"""
性能监控模块
实时监控系统性能和资源使用情况
"""
import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
import json

@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    
    # 应用层指标
    total_requests: int = 0
    active_requests: int = 0
    queue_size: int = 0
    response_time_avg: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "system": {
                "cpu_percent": self.cpu_percent,
                "memory_percent": self.memory_percent,
                "memory_used_mb": self.memory_used_mb,
                "disk_io": {
                    "read_mb": self.disk_io_read_mb,
                    "write_mb": self.disk_io_write_mb
                },
                "network": {
                    "sent_mb": self.network_sent_mb,
                    "recv_mb": self.network_recv_mb
                }
            },
            "application": {
                "total_requests": self.total_requests,
                "active_requests": self.active_requests,
                "queue_size": self.queue_size,
                "response_time_avg": self.response_time_avg,
                "error_rate": self.error_rate
            }
        }

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history: int = 1000, sample_interval: float = 1.0):
        self.max_history = max_history
        self.sample_interval = sample_interval
        self.metrics_history: deque = deque(maxlen=max_history)
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # 应用层统计
        self.app_metrics = {
            "total_requests": 0,
            "active_requests": 0,
            "queue_size": 0,
            "response_times": deque(maxlen=100),
            "error_count": 0,
            "start_time": time.time()
        }
        
        # 初始化系统监控基线
        self._init_baseline()
    
    def _init_baseline(self):
        """初始化系统监控基线"""
        try:
            self.process = psutil.Process()
            self.initial_disk_io = psutil.disk_io_counters()
            self.initial_network_io = psutil.net_io_counters()
        except Exception as e:
            print(f"⚠️ 性能监控初始化失败: {e}")
    
    async def start(self):
        """启动监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        print("📊 性能监控已启动")
    
    async def stop(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        print("📊 性能监控已停止")
    
    async def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                metrics = await self._collect_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
                
                await asyncio.sleep(self.sample_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ 性能监控异常: {e}")
                await asyncio.sleep(self.sample_interval)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        current_time = time.time()
        
        # 系统指标
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        disk_read_mb = 0
        disk_write_mb = 0
        if disk_io and self.initial_disk_io:
            disk_read_mb = (disk_io.read_bytes - self.initial_disk_io.read_bytes) / (1024 * 1024)
            disk_write_mb = (disk_io.write_bytes - self.initial_disk_io.write_bytes) / (1024 * 1024)
        
        # 网络IO
        network_io = psutil.net_io_counters()
        network_sent_mb = 0
        network_recv_mb = 0
        if network_io and self.initial_network_io:
            network_sent_mb = (network_io.bytes_sent - self.initial_network_io.bytes_sent) / (1024 * 1024)
            network_recv_mb = (network_io.bytes_recv - self.initial_network_io.bytes_recv) / (1024 * 1024)
        
        # 应用层指标
        with self._lock:
            avg_response_time = (
                sum(self.app_metrics["response_times"]) / len(self.app_metrics["response_times"])
                if self.app_metrics["response_times"] else 0.0
            )
            
            error_rate = (
                self.app_metrics["error_count"] / max(self.app_metrics["total_requests"], 1) * 100
            )
        
        return PerformanceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            total_requests=self.app_metrics["total_requests"],
            active_requests=self.app_metrics["active_requests"],
            queue_size=self.app_metrics["queue_size"],
            response_time_avg=avg_response_time,
            error_rate=error_rate
        )
    
    def record_request_start(self):
        """记录请求开始"""
        with self._lock:
            self.app_metrics["total_requests"] += 1
            self.app_metrics["active_requests"] += 1
    
    def record_request_end(self, response_time: float, success: bool = True):
        """记录请求结束"""
        with self._lock:
            self.app_metrics["active_requests"] = max(0, self.app_metrics["active_requests"] - 1)
            self.app_metrics["response_times"].append(response_time)
            if not success:
                self.app_metrics["error_count"] += 1
    
    def update_queue_size(self, size: int):
        """更新队列大小"""
        with self._lock:
            self.app_metrics["queue_size"] = size
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标"""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, duration_seconds: int = 300) -> List[PerformanceMetrics]:
        """获取历史指标"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        with self._lock:
            return [
                metric for metric in self.metrics_history
                if metric.timestamp >= cutoff_time
            ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            if not self.metrics_history:
                return {"status": "no_data"}
            
            recent_metrics = self.get_metrics_history(300)  # 最近5分钟
            
            if not recent_metrics:
                return {"status": "no_recent_data"}
            
            # 计算平均值和峰值
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            response_times = [m.response_time_avg for m in recent_metrics if m.response_time_avg > 0]
            
            uptime = time.time() - self.app_metrics["start_time"]
            
            return {
                "status": "ok",
                "uptime_seconds": uptime,
                "uptime_formatted": self._format_uptime(uptime),
                "current": self.metrics_history[-1].to_dict(),
                "statistics": {
                    "cpu": {
                        "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                        "max": max(cpu_values) if cpu_values else 0,
                        "min": min(cpu_values) if cpu_values else 0
                    },
                    "memory": {
                        "avg": sum(memory_values) / len(memory_values) if memory_values else 0,
                        "max": max(memory_values) if memory_values else 0,
                        "min": min(memory_values) if memory_values else 0
                    },
                    "response_time": {
                        "avg": sum(response_times) / len(response_times) if response_times else 0,
                        "max": max(response_times) if response_times else 0,
                        "min": min(response_times) if response_times else 0
                    }
                },
                "total_requests": self.app_metrics["total_requests"],
                "error_count": self.app_metrics["error_count"],
                "requests_per_second": self.app_metrics["total_requests"] / uptime if uptime > 0 else 0
            }
    
    def _format_uptime(self, seconds: float) -> str:
        """格式化运行时间"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}天")
        if hours > 0:
            parts.append(f"{hours}小时")
        if minutes > 0:
            parts.append(f"{minutes}分钟")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}秒")
        
        return "".join(parts)
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """导出指标到文件"""
        if filename is None:
            filename = f"performance_metrics_{int(time.time())}.json"
        
        with self._lock:
            data = {
                "export_time": time.time(),
                "app_info": {
                    "start_time": self.app_metrics["start_time"],
                    "total_requests": self.app_metrics["total_requests"],
                    "error_count": self.app_metrics["error_count"]
                },
                "metrics": [metric.to_dict() for metric in self.metrics_history]
            }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return filename
        except Exception as e:
            raise Exception(f"导出失败: {e}")

# 全局性能监控器实例
performance_monitor = PerformanceMonitor()
