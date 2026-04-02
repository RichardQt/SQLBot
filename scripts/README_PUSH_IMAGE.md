步骤 1：构建镜像并打上完整标签
docker build -t registry.cn-heyuan.aliyuncs.com/images-lsr/sqlbot_fx:v3.3.1 .
步骤 2：登录阿里云 Registry
docker login --username=richardlsr registry.cn-heyuan.aliyuncs.com
步骤 3：拉取镜像
docker pull registry.cn-heyuan.aliyuncs.com/images-lsr/sqlbot_fx:v3.3.1
步骤 4：推送镜像
docker push registry.cn-heyuan.aliyuncs.com/images-lsr/sqlbot_fx:v3.3.1
