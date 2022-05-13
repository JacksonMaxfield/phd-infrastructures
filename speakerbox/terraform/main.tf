// S3 Bucket for Storage
// Project Header: "evamaxfield-uw-equitensors"

// Terraform Configuration
provider "aws" {
  region = "us-west-2"
}

terraform {

  backend "s3" {
    bucket = "evamaxfield-uw-equitensors-terraform-state-files"
    key    = "speakerbox.prod.tfstate"
    region = "us-west-2"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~>4.13.0"
    }
  }

  required_version = "~> 1.1.9"
}


// Resources
// Bucket
resource "aws_s3_bucket" "speakerbox_storage" {
  bucket = "evamaxfield-uw-equitensors-speakerbox"
}

// Security
resource "aws_s3_bucket_acl" "speakerbox_storage_acl" {
  bucket = aws_s3_bucket.speakerbox_storage.id
  acl    = "private"
}

// Versioning
resource "aws_s3_bucket_versioning" "speakerbox_storage_versioning" {
  bucket = aws_s3_bucket.speakerbox_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}
