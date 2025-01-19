## Setup

### Setup AWS EC2 nitro instance
The notary is meant to run within an aws ec2 instance with nitro enclave enabled.

You can do it with UI (recommended) or with CLI:

``` shell
aws ec2 run-instances \
--image-id ami-0b5eea76982371e91 \
--count 1 \
--instance-type m5.xlarge \
--key-name <key-name> \
--security-group-ids sg-0a006bcdd6147509f \
--tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=my-nitro-tee-1}]' \
--enclave-options 'Enabled=true'
```

If you go with the UI:
- Pick a instance good enough such as md5.xlarge, enable nitro 
- Make sure that the ports needed by your application are open by checking the security group, in "security" tab in the instance screen.

### Setup environment

```
sudo yum install -y git
```

then clone this repository and run `./setup.sh`

[see aws doc on nitro enclaves](https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclave-cli-install.html)

If the docker image is too large, it might be necessary to allocate more memory for the enclave.

``` shell
sudo nano /etc/nitro_enclaves/allocator.yaml

sudo systemctl restart nitro-enclaves-allocator.service

```
Then update `run-enclave.sh` to request enought memory for the enclave.

## Run

Run `make` to build the TEE image and run it.

Run `./run-enclave.sh` to run the enclave without rebuilding the image.

See `Makefile` for more commands


### Run enclave in production

Run `run-enclave.sh` without the `--debug` flag to run the enclave in production mode.
The logs will not be displayed in production mode.

### Install gvproxy to open the TEE to the outside world

By default the TEE is not accessible from the outside world. 
It requires a TAP interface and a reverse proxy.
To install the networking interface, run the following commands:

```
git clone https://github.com/containers/gvisor-tap-vsock.git
cd gvisor-tap-vsock
make
```

Then run the proxy

```
sudo ./gvproxy.sh
```

## PCRS 

The PCRS are the platform configuration registers.
They are used to verify the integrity of the code running in the enclave and to ensure that it is not tampered with.

Run `nitro-cli describe-enclaves` to print the PCRS of the enclave image.

## Code attestation

The code attestation is a base64 encoded string signed by AWS that authenticates the code of the extension.
The code attestation contains the PCRS, a nonce and other information.

Once the notary is running, call the endpoint `/enclave/attestation?nonce=<nonce>` to get the code attestation, with nonce being a 40 digits hexadecimal string of your choice.

## Add new providers 

Update the config link in the `config.yml` file.

Then run `make` to run the enclave image again.

### Setup TLS

- Use certbot to get the certificates
- Copy them in a tls folder as expected by Dockerfile
- Update config.yml to enable TLS and set providers

- Run `make` to build the enclave image again- Run `make` to build the enclave image again

### Setup cert renewal
- Setup certbot deploy hook to update the certificates after renewal.
The config lives in /etc/letsencrypt/renewal

```
renew_hook = sudo cp /etc/letsencrypt/live/notary.freysa.ai/fullchain.pem  ~/notary-nitriding-deployment/tls/notary.crt && cp /etc/letsencrypt/live/notary.freysa.ai/privkey.pem  ~/notary-nitriding-deployment/tls/notary.key
```
then 
```
sudo certbot renew --dry-run
```


