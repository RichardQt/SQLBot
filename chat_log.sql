/*
 Navicat Premium Dump SQL

 Source Server         : sqlbot-pg-docker
 Source Server Type    : PostgreSQL
 Source Server Version : 170006 (170006)
 Source Host           : localhost:5432
 Source Catalog        : sqlbot
 Source Schema         : public

 Target Server Type    : PostgreSQL
 Target Server Version : 170006 (170006)
 File Encoding         : 65001

 Date: 07/11/2025 15:45:29
*/


-- ----------------------------
-- Table structure for chat_log
-- ----------------------------
DROP TABLE IF EXISTS "public"."chat_log";
CREATE TABLE "public"."chat_log" (
  "id" int8 NOT NULL GENERATED ALWAYS AS IDENTITY (
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1
),
  "type" varchar(3) COLLATE "pg_catalog"."default",
  "operate" varchar(3) COLLATE "pg_catalog"."default",
  "pid" int8,
  "ai_modal_id" int8,
  "base_modal" varchar(255) COLLATE "pg_catalog"."default",
  "messages" jsonb,
  "reasoning_content" text COLLATE "pg_catalog"."default",
  "start_time" timestamp(6),
  "finish_time" timestamp(6),
  "token_usage" jsonb
)
;

-- ----------------------------
-- Primary Key structure for table chat_log
-- ----------------------------
ALTER TABLE "public"."chat_log" ADD CONSTRAINT "chat_log_pkey" PRIMARY KEY ("id");
