# Minimal placeholders; fill with your account details for deployment.
provider "aws" {
  region = var.region
}

variable "region" { default = "us-east-1" }

resource "aws_s3_bucket" "marl_data" {
  bucket = "economic-marl-data"
  versioning { enabled = true }
}

resource "aws_s3_bucket" "marl_results" {
  bucket = "economic-marl-results"
}

versioning {
  enabled = true
}

lifecycle_rule {
  id      = "expire-noncurrent-versions"
  enabled = true

  noncurrent_version_transition {
    days          = 90
    storage_class = "STANDARD_IA"
  }

  noncurrent_version_expiration {
    days = 365
  }
}
# Add IAM roles, EventBridge schedules, and SageMaker training jobs as needed.
