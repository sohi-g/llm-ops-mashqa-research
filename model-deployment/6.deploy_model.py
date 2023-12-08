# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from google.cloud import aiplatform
SERVING_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:gptq"

model_id = "sohi-g/MASHQA-Mistral-7B-Instruct"

machine_type = "g2-standard-8"
accelerator_type = "NVIDIA_L4"
accelerator_count = 1

# Fill with the created service account.
service_account = ""

endpoint = aiplatform.Endpoint.create(display_name=f"{model_id}-endpoint")
vllm_args = [
    "--host=0.0.0.0",
    "--port=7080",
    f"--model={model_id}",
    f"--tensor-parallel-size={accelerator_count}",
    "--swap-space=16",
    "--gpu-memory-utilization=0.9",
    "--disable-log-stats",
]

model = aiplatform.Model.upload(
    display_name=model_id,
    serving_container_image_uri=SERVING_DOCKER_URI,
    serving_container_command=["python", "-m", "vllm.entrypoints.api_server"],
    serving_container_args=vllm_args,
    serving_container_ports=[7080],
    serving_container_predict_route="/generate",
    serving_container_health_route="/ping",
    
)

model.deploy(
    endpoint=endpoint,
    machine_type=machine_type,
    accelerator_type=accelerator_type,
    accelerator_count=accelerator_count,
    deploy_request_timeout=1800,
    service_account=service_account,
)
# -


