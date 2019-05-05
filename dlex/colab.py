# Imports the Google Cloud client library
from google.cloud import storage
from configs import Configs

if __name__ == "__main__":
    configs = Configs(mode="train")

    # Instantiates a client
    client = storage.Client()
    bucket = client.get_bucket('dl-torch')
