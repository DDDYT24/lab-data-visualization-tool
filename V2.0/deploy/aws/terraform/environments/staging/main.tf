module "network" {
  source = "../../modules/network"

  name_prefix              = local.name_prefix
  environment              = "staging"
  vpc_cidr                 = "10.20.0.0/16"
  public_subnet_cidrs      = ["10.20.0.0/24", "10.20.1.0/24"]
  application_subnet_cidrs = ["10.20.10.0/24", "10.20.11.0/24"]
  database_subnet_cidrs    = ["10.20.20.0/24", "10.20.21.0/24"]
}

module "security" {
  source = "../../modules/security"

  name_prefix            = local.name_prefix
  environment            = "staging"
  aws_region             = var.aws_region
  vpc_id                 = module.network.vpc_id
  vpc_cidr               = module.network.vpc_cidr
  application_subnet_ids = module.network.application_subnet_ids
}

module "data" {
  source = "../../modules/data"

  name_prefix                    = local.name_prefix
  environment                    = "staging"
  aws_region                     = var.aws_region
  vpc_id                         = module.network.vpc_id
  application_route_table_ids    = module.network.application_route_table_ids
  database_subnet_ids            = module.network.database_subnet_ids
  database_security_group_id     = module.security.database_security_group_id
  database_instance_class        = var.database_instance_class
  database_multi_az              = false
  database_deletion_protection   = true
  database_backup_retention_days = 7
  application_security_group_ids = {
    api    = module.security.api_security_group_id
    web    = module.security.web_security_group_id
    worker = module.security.worker_security_group_id
  }
}

module "edge" {
  source = "../../modules/edge"

  name_prefix                = local.name_prefix
  environment                = "staging"
  vpc_id                     = module.network.vpc_id
  public_subnet_ids          = module.network.public_subnet_ids
  alb_security_group_id      = module.security.alb_security_group_id
  domain_name                = var.domain_name
  route53_zone_id            = var.route53_zone_id
  enable_deletion_protection = true
  enable_waf                 = var.enable_waf
}

module "compute" {
  source = "../../modules/compute"

  name_prefix                     = local.name_prefix
  environment                     = "staging"
  aws_region                      = var.aws_region
  domain_name                     = var.domain_name
  application_subnet_ids          = module.network.application_subnet_ids
  api_security_group_id           = module.security.api_security_group_id
  web_security_group_id           = module.security.web_security_group_id
  worker_security_group_id        = module.security.worker_security_group_id
  api_target_group_arn            = module.edge.api_target_group_arn
  web_target_group_arn            = module.edge.web_target_group_arn
  alb_arn_suffix                  = module.edge.alb_arn_suffix
  api_target_group_arn_suffix     = module.edge.api_target_group_arn_suffix
  web_target_group_arn_suffix     = module.edge.web_target_group_arn_suffix
  trusted_proxy_cidrs             = module.network.public_subnet_cidrs
  data_bucket_name                = module.data.bucket_name
  data_bucket_arn                 = module.data.bucket_arn
  object_prefix                   = module.data.object_prefix
  object_kms_key_arn              = module.data.object_kms_key_arn
  ses_identity_name               = var.ses_identity_name
  ses_from_address                = var.ses_from_address
  ses_configuration_set_name      = "${local.name_prefix}-staging"
  api_image_digest                = var.api_image_digest
  web_image_digest                = var.web_image_digest
  activate_services               = var.activate_services
  enable_worker_delete_permission = var.enable_worker_delete_permission
}
