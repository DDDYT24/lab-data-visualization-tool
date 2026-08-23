output "vpc_id" {
  value = aws_vpc.this.id
}

output "vpc_cidr" {
  value = aws_vpc.this.cidr_block
}

output "availability_zones" {
  value = local.availability_zones
}

output "public_subnet_ids" {
  value = [for key in sort(keys(aws_subnet.public)) : aws_subnet.public[key].id]
}

output "public_subnet_cidrs" {
  value = [for key in sort(keys(aws_subnet.public)) : aws_subnet.public[key].cidr_block]
}

output "application_subnet_ids" {
  value = [for key in sort(keys(aws_subnet.application)) : aws_subnet.application[key].id]
}

output "application_route_table_ids" {
  value = [for key in sort(keys(aws_route_table.application)) : aws_route_table.application[key].id]
}

output "database_subnet_ids" {
  value = [for key in sort(keys(aws_subnet.database)) : aws_subnet.database[key].id]
}
