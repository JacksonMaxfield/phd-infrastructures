// S3 Bucket for Storage

provider "aws" {
  region = "us-west-2"
}

terraform {

  backend "s3" {
    bucket = "evamaxfield-uw-equitensors-terraform-state-files"
    key    = "speakerbox.prod.tfstate"
    region = "us-west-2"

    versioning {
      enabled = true
    }
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~>4.13.0"
    }
  }

  required_version = "~> 1.1.9"

}
