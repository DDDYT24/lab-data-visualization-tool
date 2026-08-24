output "bucket_name" {
  value = aws_s3_bucket.data.id
}

output "bucket_arn" {
  value = aws_s3_bucket.data.arn
}

output "object_prefix" {
  value = var.object_prefix
}

output "object_kms_key_arn" {
  value = aws_kms_key.objects.arn
}

output "database_address" {
  value = aws_db_instance.postgres.address
}

output "database_arn" {
  value = aws_db_instance.postgres.arn
}

output "database_identifier" {
  value = aws_db_instance.postgres.identifier
}

output "database_subnet_group_name" {
  value = aws_db_subnet_group.this.name
}

output "database_port" {
  value = aws_db_instance.postgres.port
}

output "database_master_secret_arn" {
  value     = aws_db_instance.postgres.master_user_secret[0].secret_arn
  sensitive = true
}

output "database_kms_key_arn" {
  value = aws_kms_key.database.arn
}
