try:
    from openenv.core.env_server import create_app
except ImportError:
    from core.env_server import create_app

from models import CustomerSupportAction, CustomerSupportObservation
from server.customer_support_environment import CustomerSupportEnvironment

app = create_app(
    CustomerSupportEnvironment,
    CustomerSupportAction,
    CustomerSupportObservation,
    env_name="customer_support_env",
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
