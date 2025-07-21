"""
用户会话管理服务
管理用户连接、认证和会话状态
"""
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseService, ServiceStatus

class UserStatus(Enum):
    """用户状态"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"

@dataclass
class UserSession:
    """用户会话"""
    user_id: str
    session_id: str
    status: UserStatus = UserStatus.ONLINE
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    total_requests: int = 0
    active_requests: int = 0
    max_concurrent_requests: int = 5
    request_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self):
        """更新活动时间"""
        self.last_activity = time.time()
    
    def is_expired(self, timeout: int = 3600) -> bool:
        """检查会话是否过期"""
        return time.time() - self.last_activity > timeout
    
    def can_accept_request(self) -> bool:
        """检查是否可以接受新请求"""
        return self.active_requests < self.max_concurrent_requests

@dataclass
class UserQuota:
    """用户配额"""
    user_id: str
    daily_limit: int = 1000
    hourly_limit: int = 100
    concurrent_limit: int = 5
    daily_used: int = 0
    hourly_used: int = 0
    last_reset_daily: float = field(default_factory=time.time)
    last_reset_hourly: float = field(default_factory=time.time)
    
    def reset_daily_if_needed(self):
        """如果需要重置每日配额"""
        current_time = time.time()
        if current_time - self.last_reset_daily >= 86400:  # 24小时
            self.daily_used = 0
            self.last_reset_daily = current_time
    
    def reset_hourly_if_needed(self):
        """如果需要重置每小时配额"""
        current_time = time.time()
        if current_time - self.last_reset_hourly >= 3600:  # 1小时
            self.hourly_used = 0
            self.last_reset_hourly = current_time
    
    def can_make_request(self) -> bool:
        """检查是否可以发起请求"""
        self.reset_daily_if_needed()
        self.reset_hourly_if_needed()
        return (self.daily_used < self.daily_limit and 
                self.hourly_used < self.hourly_limit)
    
    def consume_quota(self):
        """消费配额"""
        self.daily_used += 1
        self.hourly_used += 1

class UserSessionManager(BaseService):
    """用户会话管理服务"""
    
    def __init__(self, service_id: Optional[str] = None):
        super().__init__(service_id)
        self.sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self.quotas: Dict[str, UserQuota] = {}
        self.session_timeout = 3600  # 1小时
        self.cleanup_interval = 300  # 5分钟
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """初始化会话管理服务"""
        try:
            self.status = ServiceStatus.READY
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            return True
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.metrics.last_error = str(e)
            return False
    
    async def create_session(self, user_id: str, metadata: Dict[str, Any] = None) -> UserSession:
        """创建用户会话"""
        async with self._lock:
            session_id = str(uuid.uuid4())
            
            session = UserSession(
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {}
            )
            
            self.sessions[session_id] = session
            
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
            
            # 创建用户配额（如果不存在）
            if user_id not in self.quotas:
                self.quotas[user_id] = UserQuota(user_id=user_id)
            
            return session
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session and not session.is_expired(self.session_timeout):
            session.update_activity()
            return session
        elif session:
            # 会话过期，清理
            await self._remove_session(session_id)
        return None
    
    async def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """获取用户的所有会话"""
        session_ids = self.user_sessions.get(user_id, set())
        sessions = []
        
        for session_id in list(session_ids):  # 创建副本避免修改冲突
            session = self.sessions.get(session_id)
            if session and not session.is_expired(self.session_timeout):
                sessions.append(session)
            elif session:
                await self._remove_session(session_id)
        
        return sessions
    
    async def remove_session(self, session_id: str) -> bool:
        """移除会话"""
        return await self._remove_session(session_id)
    
    async def _remove_session(self, session_id: str) -> bool:
        """内部移除会话方法"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                user_id = session.user_id
                del self.sessions[session_id]
                
                if user_id in self.user_sessions:
                    self.user_sessions[user_id].discard(session_id)
                    if not self.user_sessions[user_id]:
                        del self.user_sessions[user_id]
                
                return True
            return False
    
    async def update_session_status(self, session_id: str, status: UserStatus) -> bool:
        """更新会话状态"""
        session = await self.get_session(session_id)
        if session:
            session.status = status
            session.update_activity()
            return True
        return False
    
    async def check_quota(self, user_id: str) -> bool:
        """检查用户配额"""
        if user_id not in self.quotas:
            self.quotas[user_id] = UserQuota(user_id=user_id)
        
        quota = self.quotas[user_id]
        return quota.can_make_request()
    
    async def consume_quota(self, user_id: str) -> bool:
        """消费用户配额"""
        if not await self.check_quota(user_id):
            return False
        
        quota = self.quotas[user_id]
        quota.consume_quota()
        return True
    
    async def get_quota_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取配额信息"""
        if user_id not in self.quotas:
            return None
        
        quota = self.quotas[user_id]
        quota.reset_daily_if_needed()
        quota.reset_hourly_if_needed()
        
        return {
            "daily_limit": quota.daily_limit,
            "daily_used": quota.daily_used,
            "daily_remaining": quota.daily_limit - quota.daily_used,
            "hourly_limit": quota.hourly_limit,
            "hourly_used": quota.hourly_used,
            "hourly_remaining": quota.hourly_limit - quota.hourly_used,
            "concurrent_limit": quota.concurrent_limit
        }
    
    async def increment_active_requests(self, session_id: str) -> bool:
        """增加活跃请求数"""
        session = await self.get_session(session_id)
        if session and session.can_accept_request():
            session.active_requests += 1
            session.total_requests += 1
            return True
        return False
    
    async def decrement_active_requests(self, session_id: str):
        """减少活跃请求数"""
        session = await self.get_session(session_id)
        if session:
            session.active_requests = max(0, session.active_requests - 1)
    
    async def _cleanup_loop(self):
        """定期清理过期会话"""
        while self.status != ServiceStatus.STOPPED:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"会话清理任务异常: {e}")
    
    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired(self.session_timeout):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self._remove_session(session_id)
        
        if expired_sessions:
            print(f"清理了 {len(expired_sessions)} 个过期会话")
    
    async def process(self, request: Any) -> Any:
        """处理请求（基类接口实现）"""
        # 可以处理会话相关的请求
        return {"status": "ok", "message": "Session manager is ready"}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        total_sessions = len(self.sessions)
        total_users = len(self.user_sessions)
        active_sessions = sum(1 for s in self.sessions.values() 
                            if s.status == UserStatus.ONLINE)
        
        return {
            "status": self.status.value,
            "service_id": self.service_id,
            "total_sessions": total_sessions,
            "total_users": total_users,
            "active_sessions": active_sessions,
            "session_timeout": self.session_timeout,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "active_requests": self.metrics.active_requests,
                "error_count": self.metrics.error_count
            }
        }
    
    async def shutdown(self):
        """关闭服务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await super().shutdown()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        user_stats = {}
        for user_id, session_ids in self.user_sessions.items():
            sessions = [self.sessions[sid] for sid in session_ids if sid in self.sessions]
            user_stats[user_id] = {
                "session_count": len(sessions),
                "total_requests": sum(s.total_requests for s in sessions),
                "active_requests": sum(s.active_requests for s in sessions)
            }
        
        return {
            "total_users": len(self.user_sessions),
            "total_sessions": len(self.sessions),
            "user_statistics": user_stats
        }
