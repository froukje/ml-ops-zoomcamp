#################
# Deployment for Q4
#################

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

DeploymentSpec(    
    name="cron-schedule-deployment",
    flow_location="homework.py",
    schedule=CronSchedule(
        cron="0 9 15 * *")
)

