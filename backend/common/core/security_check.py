"""
安全检查模块
用于检测不安全的配置，特别是默认密钥
"""
import sys
import logging
from typing import List

logger = logging.getLogger(__name__)

# 已知的不安全默认密钥列表
KNOWN_INSECURE_KEYS = [
    "y5txe1mRmS_JpOrUzFzHEu-kIQn3lf7ll0AOv9DQh0s",
    "pu1-jV5HncFh0t3UGe_IAF8nuEqWvENJm36eB_ESWoQ",
]

# 弱密钥特征
WEAK_KEY_PATTERNS = [
    "test",
    "demo",
    "example",
    "default",
    "password",
    "secret",
    "key",
    "admin",
]


def check_secret_key_security(secret_key: str) -> tuple[bool, List[str]]:
    """
    检查SECRET_KEY的安全性
    
    Args:
        secret_key: 要检查的密钥
        
    Returns:
        tuple: (is_secure: bool, warnings: List[str])
    """
    warnings = []
    
    # 检查是否为已知的不安全密钥
    if secret_key in KNOWN_INSECURE_KEYS:
        warnings.append(
            f"严重安全警告: 检测到使用默认密钥！这是一个已知的公开密钥，"
            f"攻击者可以使用此密钥伪造JWT令牌并绕过身份验证。"
        )
        return False, warnings
    
    # 检查密钥长度
    if len(secret_key) < 32:
        warnings.append(
            f"安全警告: SECRET_KEY长度过短 (当前: {len(secret_key)} 字符)，"
            f"建议至少32字符以上。"
        )
        return False, warnings
    
    # 检查是否包含弱密钥特征
    secret_key_lower = secret_key.lower()
    for pattern in WEAK_KEY_PATTERNS:
        if pattern in secret_key_lower:
            warnings.append(
                f"安全警告: SECRET_KEY包含弱特征词 '{pattern}'，"
                f"建议使用完全随机生成的密钥。"
            )
            return False, warnings
    
    # 检查密钥的随机性（简单检查：是否有足够的字符种类）
    has_upper = any(c.isupper() for c in secret_key)
    has_lower = any(c.islower() for c in secret_key)
    has_digit = any(c.isdigit() for c in secret_key)
    has_special = any(not c.isalnum() for c in secret_key)
    
    char_types = sum([has_upper, has_lower, has_digit, has_special])
    if char_types < 3:
        warnings.append(
            "安全提示: SECRET_KEY的复杂度较低，建议包含大小写字母、数字和特殊字符。"
        )
    
    return True, warnings


def enforce_security_check(secret_key: str, strict_mode: bool = True) -> None:
    """
    强制执行安全检查
    
    Args:
        secret_key: 要检查的密钥
        strict_mode: 严格模式，如果检测到不安全的密钥则终止程序
    """
    is_secure, warnings = check_secret_key_security(secret_key)
    
    if not is_secure:
        logger.error("=" * 80)
        logger.error("检测到安全配置问题:")
        for warning in warnings:
            logger.error(f"  - {warning}")
        
        if strict_mode:
            logger.error("")
            logger.error("修复方法:")
            logger.error("  1. 生成新的随机密钥:")
            logger.error("     python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
            logger.error("")
            logger.error("  2. 在 .env 文件中设置 SECRET_KEY 环境变量:")
            logger.error("     SECRET_KEY=<your-generated-key>")
            logger.error("")
            logger.error("  3. 如果使用 Docker，请在 docker-compose.yaml 中设置:")
            logger.error("     environment:")
            logger.error("       SECRET_KEY: <your-generated-key>")
            logger.error("")
            logger.error("为了您的系统安全，程序将终止运行。")
            logger.error("=" * 80)
            sys.exit(1)
        else:
            logger.warning("=" * 80)
            logger.warning("检测到安全配置问题，但未启用严格模式")
            for warning in warnings:
                logger.warning(f"  - {warning}")
            logger.warning("强烈建议尽快修复！")
            logger.warning("=" * 80)
    elif warnings:
        for warning in warnings:
            logger.info(warning)
    else:
        logger.info("SECRET_KEY 安全检查通过")


def get_security_recommendations() -> List[str]:
    """
    获取安全建议
    
    Returns:
        List[str]: 安全建议列表
    """
    return [
        "使用 secrets.token_urlsafe(32) 或更长的密钥",
        "每个部署环境使用不同的 SECRET_KEY",
        "定期轮换 SECRET_KEY（建议每3-6个月）",
        "将 SECRET_KEY 存储在环境变量或密钥管理服务中",
        "切勿将 SECRET_KEY 提交到版本控制系统",
        "使用 HTTPS 保护传输中的 JWT 令牌",
        "设置合理的令牌过期时间（建议不超过24小时）",
        "实现令牌刷新机制",
        "记录和监控异常的身份验证尝试",
        "考虑使用 RS256 等非对称加密算法（适用于大型分布式系统）",
    ]
