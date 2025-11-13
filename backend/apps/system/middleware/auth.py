
import base64
import json
from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse
import jwt
from sqlmodel import Session
from starlette.middleware.base import BaseHTTPMiddleware
from apps.system.models.system_model import AssistantModel
from common.core.db import engine 
from apps.system.crud.assistant import get_assistant_info, get_assistant_user
from apps.system.crud.user import get_user_by_account, get_user_info
from apps.system.schemas.system_schema import AssistantHeader, UserInfoDTO
from common.core import security
from common.core.config import settings
from common.core.schemas import TokenPayload
from common.utils.locale import I18n
from common.utils.utils import SQLBotLogUtil
from common.utils.whitelist import whiteUtils
from fastapi.security.utils import get_authorization_scheme_param
from common.core.deps import get_i18n
class TokenMiddleware(BaseHTTPMiddleware):
    
    
    
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        
        if self.is_options(request) or whiteUtils.is_whitelisted(request.url.path):
            return await call_next(request)
        assistantTokenKey = settings.ASSISTANT_TOKEN_KEY
        assistantToken = request.headers.get(assistantTokenKey)
        trans = await get_i18n(request)
        #if assistantToken and assistantToken.lower().startswith("assistant "):
        if assistantToken:
            validator: tuple[any] = await self.validateAssistant(assistantToken, trans)
            if validator[0]:
                request.state.current_user = validator[1]
                request.state.assistant = validator[2]
                return await call_next(request)
            message = trans('i18n_permission.authenticate_invalid', msg = validator[1])
            return JSONResponse(message, status_code=401, headers={"Access-Control-Allow-Origin": "*"})
        #validate pass
        tokenkey = settings.TOKEN_KEY
        token = request.headers.get(tokenkey)
        validate_pass, data = await self.validateToken(token, trans)
        if validate_pass:
            request.state.current_user = data
            return await call_next(request)
        
        message = trans('i18n_permission.authenticate_invalid', msg = data)
        return JSONResponse(message, status_code=401, headers={"Access-Control-Allow-Origin": "*"})
    
    def is_options(self, request: Request):
        return request.method == "OPTIONS"
    
    async def validateToken(self, token: Optional[str], trans: I18n):
        if not token:
            return False, f"Miss Token[{settings.TOKEN_KEY}]!"
        schema, param = get_authorization_scheme_param(token)
        if schema.lower() != "bearer":
            return False, f"Token schema error!"
        try: 
            # 严格验证JWT签名和过期时间
            payload = jwt.decode(
                param, 
                settings.SECRET_KEY, 
                algorithms=[security.ALGORITHM],
                options={
                    "verify_signature": True,  # 强制验证签名
                    "verify_exp": True,         # 强制验证过期时间
                    "require": ["exp", "id"]    # 必须包含exp和id字段
                }
            )
            token_data = TokenPayload(**payload)
            
            # 验证token中的用户ID是否有效
            if not token_data.id or token_data.id <= 0:
                return False, "Invalid user ID in token"
            
            with Session(engine) as session:
                session_user = await get_user_info(session = session, user_id = token_data.id)
                if not session_user:
                    message = trans('i18n_not_exist', msg = trans('i18n_user.account'))
                    raise Exception(message)
                session_user = UserInfoDTO.model_validate(session_user)
                if session_user.status != 1:
                    message = trans('i18n_login.user_disable', msg = trans('i18n_concat_admin'))
                    raise Exception(message)
                if not session_user.oid or session_user.oid == 0:
                    message = trans('i18n_login.no_associated_ws', msg = trans('i18n_concat_admin'))
                    raise Exception(message)
                
                # 验证token中的用户信息与数据库一致
                if token_data.id != session_user.id:
                    return False, "Token user ID mismatch"
                    
                return True, session_user
        except jwt.ExpiredSignatureError:
            SQLBotLogUtil.warning(f"Token expired")
            return False, trans('i18n_permission.token_expired')
        except jwt.InvalidSignatureError:
            SQLBotLogUtil.warning(f"Invalid token signature")
            return False, "Invalid token signature"
        except jwt.DecodeError:
            SQLBotLogUtil.warning(f"Token decode error")
            return False, "Token decode error"
        except Exception as e:
            msg = str(e)
            SQLBotLogUtil.exception(f"Token validation error: {msg}")
            return False, e
            
    
    async def validateAssistant(self, assistantToken: Optional[str], trans: I18n) -> tuple[any]:
        if not assistantToken:
            return False, f"Miss Token[{settings.TOKEN_KEY}]!"
        schema, param = get_authorization_scheme_param(assistantToken)
        
        
        try:
            if schema.lower() == 'embedded':
                return await self.validateEmbedded(param, trans)
            if schema.lower() != "assistant":
                return False, f"Token schema error!" 
            
            # 严格验证JWT签名和过期时间
            payload = jwt.decode(
                param, 
                settings.SECRET_KEY, 
                algorithms=[security.ALGORITHM],
                options={
                    "verify_signature": True,  # 强制验证签名
                    "verify_exp": True,         # 强制验证过期时间
                    "require": ["exp", "id", "assistant_id"]  # 必须包含这些字段
                }
            )
            token_data = TokenPayload(**payload)
            
            # 验证必要字段
            if not payload.get('assistant_id'):
                return False, f"Missing assistant_id in token payload"
            if not token_data.id or token_data.id <= 0:
                return False, "Invalid user ID in token"
                
            with Session(engine) as session:
                session_user = get_assistant_user(id = token_data.id)
                assistant_info = await get_assistant_info(session=session, assistant_id=payload['assistant_id'])
                
                if not assistant_info:
                    return False, "Assistant not found"
                    
                assistant_info = AssistantModel.model_validate(assistant_info)
                assistant_info = AssistantHeader.model_validate(assistant_info.model_dump(exclude_unset=True))
                
                if assistant_info and assistant_info.type == 0:
                    if payload.get('oid'):
                        session_user.oid = int(payload['oid'])
                    else:
                        assistant_oid = 1
                        configuration = assistant_info.configuration
                        config_obj = json.loads(configuration) if configuration else {}
                        assistant_oid = config_obj.get('oid', 1)
                        session_user.oid = int(assistant_oid)
                        
                return True, session_user, assistant_info
        except jwt.ExpiredSignatureError:
            SQLBotLogUtil.warning(f"Assistant token expired")
            return False, trans('i18n_permission.token_expired')
        except jwt.InvalidSignatureError:
            SQLBotLogUtil.warning(f"Invalid assistant token signature")
            return False, "Invalid token signature"
        except jwt.DecodeError:
            SQLBotLogUtil.warning(f"Assistant token decode error")
            return False, "Token decode error"
        except Exception as e:
            SQLBotLogUtil.exception(f"Assistant validation error: {str(e)}")
            return False, e
    
    async def validateEmbedded(self, param: str, trans: I18n) -> tuple[any]:
        try: 
            # 安全修复: 启用签名验证和过期时间验证
            # 移除 verify_signature=False 以防止JWT伪造攻击
            payload: dict = jwt.decode(
                param,
                settings.SECRET_KEY,
                algorithms=[security.ALGORITHM],
                options={
                    "verify_signature": True,   # 强制验证签名 - 安全关键
                    "verify_exp": True,          # 强制验证过期时间
                    "require": ["account"]       # 必须包含account字段
                }
            )
            
            # 验证签发者（如果存在）
            if "iss" in payload and payload["iss"] != "SQLBot":
                SQLBotLogUtil.warning(f"Token issuer mismatch: {payload.get('iss')}")
                return False, "Invalid token issuer"
            
            app_key = payload.get('appId', '')
            embeddedId = payload.get('embeddedId', None)
            if not embeddedId:
                embeddedId = xor_decrypt(app_key)
            
            if not payload.get('account'):
                return False, f"Missing account in token payload"
                
            account = payload['account']
            
            with Session(engine) as session:
                session_user = get_user_by_account(session = session, account=account)
                if not session_user:
                    message = trans('i18n_not_exist', msg = trans('i18n_user.account'))
                    raise Exception(message)
                    
                session_user = await get_user_info(session = session, user_id = session_user.id)
                
                session_user = UserInfoDTO.model_validate(session_user)
                if session_user.status != 1:
                    message = trans('i18n_login.user_disable', msg = trans('i18n_concat_admin'))
                    raise Exception(message)
                if not session_user.oid or session_user.oid == 0:
                    message = trans('i18n_login.no_associated_ws', msg = trans('i18n_concat_admin'))
                    raise Exception(message)
                    
                assistant_info = await get_assistant_info(session=session, assistant_id=embeddedId)
                if not assistant_info:
                    return False, "Embedded assistant not found"
                    
                assistant_info = AssistantModel.model_validate(assistant_info)
                assistant_info = AssistantHeader.model_validate(assistant_info.model_dump(exclude_unset=True))
                return True, session_user, assistant_info
                
        except jwt.ExpiredSignatureError:
            SQLBotLogUtil.warning(f"Embedded token expired")
            return False, trans('i18n_permission.token_expired')
        except jwt.InvalidSignatureError:
            SQLBotLogUtil.warning(f"Invalid embedded token signature - possible forgery attempt")
            return False, "Invalid token signature"
        except jwt.DecodeError:
            SQLBotLogUtil.warning(f"Embedded token decode error")
            return False, "Token decode error"
        except Exception as e:
            SQLBotLogUtil.exception(f"Embedded validation error: {str(e)}")
            return False, e
    
def xor_decrypt(encrypted_str: str, key: int = 0xABCD1234) -> int:
    encrypted_bytes = base64.urlsafe_b64decode(encrypted_str)
    hex_str = encrypted_bytes.hex()
    encrypted_num = int(hex_str, 16)
    return encrypted_num ^ key