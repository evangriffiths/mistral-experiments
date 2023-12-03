# Vast.ai 'docker in docker' debug

On a defual vast.ai linux container

```bash
$ sudo docker run hello-world
docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?.
See 'docker run --help'.
```

## Workaround 1

Create AWS EC2 instance (uses VM instead of docker container). SSH in/connect using web gui terminal.

```bash
sudo yum install -y docker
sudo systemctl start docker
sudo docker run hello-world
```

Downside: ~3x more expensive (for slower 24GB GPU). 1.2 vs $0.4 $/hr.

## Workaround 2(?)

Vast.ai supports docker image templates. Could create and push helix docker image and use as the template.
