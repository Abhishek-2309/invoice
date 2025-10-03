# Invoice Processing Pipeline (Qwen3 + Nanonets-OCR-s via vLLM)

This pipeline runs two LLMs on a single GPU server. The main difference between this pipeline and the previous pipeline is the usage of the vLLM library to serve the 2nd model. vLLM provides a high throughput and quicker inference based on the concept of paged attention.

Three containers were created - one for the entire application along with its endpoints , second for the vLLM served Qwen3 model and another for the vLLM served OCR model . When a request is made for a file, OCR processing is done in the OCR container and a request made to the vLLM container to provide an output based on this. These are the containers:

* OCR LLM (vllm-ocr): nanonets/Nanonets-OCR-s served through vLLM (for extracting text + tables as markdown/HTML from PDFs/images).
* Text LLM (vllm): Qwen3-8B served through vLLM (for parsing OCR markdown into structured JSON).
* App Service (app): FastAPI app that exposes REST endpoints (/upload, /upload_zip) and internally orchestrates OCR + text parsing.

This is deployed on AWS g6e.xlarge (1× NVIDIA L40S GPU with ~48 GB VRAM, 64 GB CPU RAM). with a Disk Size of: 120 GB (models + cache + Docker layers).

The reason for running Docker + vLLM is because
* vLLM enables high-throughput inference via paged attention, cutting down processing time
* Docker isolates all dependencies and CUDA versions. This enables us to easily fix library mismatches and version dependencies.

# Steps to Set Up

# 1. Launch GPU instance and create the files.

Since 2 LLMs are run with a storage close to 30-35GB atleast combined, it is advisable to run it on an instance like g6e.xlarge, which has 48gb GPU VRAM as well as Nvidia’s L40s GPU which provides faster throughput compared to most other chipsets.
* Use AWS g6e.xlarge (Ubuntu 24.04).
* Attach 100–120 GB disk.

This setup also requires a docker file, a docker yml file and an env file for setting up the vLLM configs. These files can be modified depending on the instance and models and are needed to setup the containers.


# 2. Install NVIDIA drivers

After entering the instance, run these commands to download the required drivers.

- sudo apt update
- sudo apt -y install ubuntu-drivers-common
- sudo ubuntu-drivers install   # installs recommended, e.g. nvidia-driver-550+
- sudo reboot

After reboot:
- nvidia-smi
(Should show NVIDIA driver + L40S GPU.)

# 3. Install Docker Engine & Compose

Next, the docker engine has to be installed based on the following steps:

- sudo apt remove -y docker docker-engine docker.io containerd runc || true

- sudo apt update
- sudo apt -y install ca-certificates curl gnupg
- sudo install -m 0755 -d /etc/apt/keyrings
- curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
- echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu noble stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

- sudo apt update
- sudo apt -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

- sudo usermod -aG docker $USER
- newgrp docker

# 4. Install NVIDIA Container Toolkit

Next, the NVIDIA Container Toolkit is to be installed

- curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
   | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

- curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

- sudo apt update
- sudo apt install -y nvidia-container-toolkit

- sudo nvidia-ctk runtime configure --runtime=docker
- sudo systemctl restart docker

Check:
- docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

Should show your GPU inside Docker. Ubuntu22.04 is used as a lot of libraries have pre built wheels for it and setup is far less complicated compared to the 24.04 instance.

# 5. Clone repository

- git clone https://github.com/<you>/<your-repo>.git
- cd <your-repo>/Invoice_App

# 6. Configure environment

- cp .env.example .env
- nano .env

**Refer to the env file given with the code and enter the following**

Set:

- VLLM_API_KEY=changeme

* This is for the Qwen3 vLLM *
- VLLM_MODEL=Qwen/Qwen3-8B-Instruct-AWQ
- VLLM_MEM_UTIL=0.65
- VLLM_MAX_MODEL_LEN=16384
- VLLM_MAX_NUM_SEQS=32
- VLLM_MAX_BATCHED_TOKENS=4096

* This is for the OCR vLLM *
- OCR_VLLM_MODEL=nanonets/Nanonets-OCR-s
- OCR_VLLM_MEM_UTIL=0.30
- OCR_VLLM_MAX_MODEL_LEN=8192
- OCR_VLLM_MAX_NUM_SEQS=8
- OCR_VLLM_MAX_BATCHED_TOKENS=2048
- OCR_VLLM_BASE_URL=http://vllm-ocr:8002/v1

Here, I assigned 65% of GPU utilization and 30% to Qwen3:8B and Nanonets respectively. This is catered to the g6e instance which has 48GB of GPU VRAM and can be suitably changed for another instance.

# 7. Start containers
Always start in this order (so VRAM is sliced correctly):

- cd ~/invoice/Invoice_App
- docker compose down

* 1) Start OCR vLLM * 
- docker compose up -d vllm-ocr
- docker compose logs -f vllm-ocr   # wait until "Application startup complete."

* 2) Start text vLLM *
- docker compose up -d vllm
- docker compose logs -f vllm  # wait until "Application startup complete."

* 3) Build and start the app *
- docker compose build --no-cache app
- docker compose up -d app
- docker compose logs -f app # wait until "Application startup complete."

# 8. Verify

* GPU split:
- nvidia-smi
Two vLLM processes, one ~65% VRAM (Qwen3), one ~30% VRAM (OCR).

# 9. Send a test file
From your local machine:

curl -F "file=@/path/to/invoice.pdf" http://<EC2_PUBLIC_IP>:8080/upload

The app:
1. Calls OCR vLLM (nanonets-ocr-s) to extract text/tables → markdown.
2. Sends markdown to Qwen3 vLLM for JSON extraction.
3. Returns validated JSON response.

# 10. Maintenance
* Stop stack: docker compose down
* Free space: docker system prune -af && docker volume prune -f
* Check logs: docker compose logs -f app
