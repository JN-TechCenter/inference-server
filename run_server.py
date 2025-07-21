"""
服务器启动脚本
"""
import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置环境变量
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)

async def main():
    """主函数"""
    try:
        # 导入应用
        from app.main import app
        import uvicorn
        
        print("🚀 启动YOLO推理服务器...")
        
        # 运行服务器
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False,
            workers=1
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
