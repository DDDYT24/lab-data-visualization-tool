output "alb_security_group_id" {
  value = aws_security_group.alb.id
}

output "api_security_group_id" {
  value = aws_security_group.api.id
}

output "web_security_group_id" {
  value = aws_security_group.web.id
}

output "worker_security_group_id" {
  value = aws_security_group.worker.id
}

output "database_security_group_id" {
  value = aws_security_group.database.id
}

output "interface_endpoint_ids" {
  value = { for service, endpoint in aws_vpc_endpoint.interface : service => endpoint.id }
}
