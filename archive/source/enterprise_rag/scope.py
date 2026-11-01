"""统一证据边界：权限、命名空间、生效时点与发布快照。

所有召回、图扩展、上下文、引用和调试轨迹都必须使用同一个 EvidenceScope。
用户身份只来自服务端认证上下文，模型输出、记忆或会话文本不能改变授权。
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from app.domain.user.models import UserContext
from enterprise_rag import config


@dataclass(frozen=True)
class EvidenceScope:
    user: UserContext
    namespace: str = config.RAG_NAMESPACE
    as_of: str | None = None
    snapshot: str = ""
    acl_version: str = ""
    strict_effective_date: bool = False
    comparison: bool = False

    @classmethod
    def from_request(
        cls,
        user: UserContext,
        namespace: str | None = None,
        as_of: str | None = None,
        strict_effective_date: bool = False,
        comparison: bool = False,
        snapshot: str = "",
        acl_version: str = "",
    ) -> "EvidenceScope":
        return cls(
            user=user,
            namespace=namespace or config.RAG_NAMESPACE,
            as_of=as_of,
            snapshot=snapshot,
            acl_version=acl_version,
            strict_effective_date=bool(strict_effective_date),
            comparison=bool(comparison),
        )

    # ---- 权限 ---------------------------------------------------------
    def allows(self, chunk: dict) -> bool:
        """块级 ACL；企业语料默认全员可见，department 仅作元数据。"""
        if chunk.get("namespace") and chunk["namespace"] != self.namespace:
            return False
        user = self.user
        if user.is_admin:
            return True
        visibility = chunk.get("visibility") or "all"
        if visibility == "all":
            return True
        if visibility == "manager":
            return user.is_manager
        if visibility == "hr":
            return user.is_hr
        if visibility == "finance":
            return user.role == "finance"
        if visibility == "admin":
            return False
        return False

    def allows_future(self, chunk: dict) -> bool:
        """显式 as_of 或制度生效日问题都不得看到生效日之后的版本。"""
        if not (self.as_of and self.strict_effective_date):
            return True
        effective = chunk.get("effective_date")
        return not effective or str(effective) <= self.as_of

    def allows_effective(self, chunk: dict) -> bool:
        return self.allows(chunk) and self.allows_future(chunk)

    # ---- 版本族 -------------------------------------------------------
    def _family_ok(self, chunk: dict, index) -> bool:
        if self.comparison:
            return True
        doc_id = chunk.get("doc_id")
        family = chunk.get("doc_family")
        if not family or family not in getattr(index, "versioned_families", set()):
            return True
        allowed = index.allowed_doc_ids(self.as_of)
        return allowed is None or doc_id in allowed

    def filter(self, chunks: Iterable[dict], index=None) -> list[dict]:
        selected = []
        for chunk in chunks:
            if not self.allows_effective(chunk):
                continue
            if index is not None and not self._family_ok(chunk, index):
                continue
            selected.append(chunk)
        return selected

    def legal_chunk_ids(self, index) -> set[str]:
        return {
            chunk["chunk_id"]
            for chunk in self.filter(index.chunks, index)
            if chunk.get("chunk_id")
        }

    def legal_doc_ids(self, index) -> set[str]:
        return {
            chunk["doc_id"]
            for chunk in self.filter(index.chunks, index)
            if chunk.get("doc_id")
        }

    def allows_document(self, doc_id: str, index) -> bool:
        return doc_id in self.legal_doc_ids(index)

    # ---- 指纹 ---------------------------------------------------------
    def fingerprint(self) -> str:
        raw = "|".join(
            (
                self.user.user_id,
                self.user.role,
                self.user.department or "",
                self.namespace,
                self.as_of or "",
                self.snapshot,
                self.acl_version,
                str(self.strict_effective_date),
                str(self.comparison),
            )
        )
        return sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "user_id": self.user.user_id,
            "role": self.user.role,
            "department": self.user.department,
            "namespace": self.namespace,
            "as_of": self.as_of,
            "snapshot": self.snapshot,
            "acl_version": self.acl_version,
            "strict_effective_date": self.strict_effective_date,
            "comparison": self.comparison,
            "fingerprint": self.fingerprint(),
        }
