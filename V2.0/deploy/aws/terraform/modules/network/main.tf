data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  availability_zones = slice(sort(data.aws_availability_zones.available.names), 0, 2)
  subnet_slots = {
    for index, availability_zone in local.availability_zones : tostring(index) => {
      availability_zone = availability_zone
      public_cidr       = var.public_subnet_cidrs[index]
      application_cidr  = var.application_subnet_cidrs[index]
      database_cidr     = var.database_subnet_cidrs[index]
    }
  }
}

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.name_prefix}-${var.environment}"
  }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.name_prefix}-${var.environment}"
  }
}

resource "aws_subnet" "public" {
  for_each = local.subnet_slots

  vpc_id                  = aws_vpc.this.id
  availability_zone       = each.value.availability_zone
  cidr_block              = each.value.public_cidr
  map_public_ip_on_launch = false

  tags = {
    Name = "${var.name_prefix}-${var.environment}-public-${each.value.availability_zone}"
    Tier = "public-alb"
  }
}

resource "aws_subnet" "application" {
  for_each = local.subnet_slots

  vpc_id                  = aws_vpc.this.id
  availability_zone       = each.value.availability_zone
  cidr_block              = each.value.application_cidr
  map_public_ip_on_launch = false

  tags = {
    Name = "${var.name_prefix}-${var.environment}-application-${each.value.availability_zone}"
    Tier = "private-application"
  }
}

resource "aws_subnet" "database" {
  for_each = local.subnet_slots

  vpc_id                  = aws_vpc.this.id
  availability_zone       = each.value.availability_zone
  cidr_block              = each.value.database_cidr
  map_public_ip_on_launch = false

  tags = {
    Name = "${var.name_prefix}-${var.environment}-database-${each.value.availability_zone}"
    Tier = "isolated-database"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.name_prefix}-${var.environment}-public"
  }
}

resource "aws_route" "public_internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.this.id
}

resource "aws_route_table_association" "public" {
  for_each = aws_subnet.public

  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "application" {
  for_each = local.subnet_slots

  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.name_prefix}-${var.environment}-application-${each.value.availability_zone}"
  }
}

resource "aws_route_table_association" "application" {
  for_each = aws_subnet.application

  subnet_id      = each.value.id
  route_table_id = aws_route_table.application[each.key].id
}

resource "aws_route_table" "database" {
  for_each = local.subnet_slots

  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.name_prefix}-${var.environment}-database-${each.value.availability_zone}"
  }
}

resource "aws_route_table_association" "database" {
  for_each = aws_subnet.database

  subnet_id      = each.value.id
  route_table_id = aws_route_table.database[each.key].id
}
