from src.skills.registry import get_registry
from src.skills.builtin.calculator import CalculatorSkill
from src.skills.builtin.file_ops import FileOpsSkill, DateTimeSkill, SystemInfoSkill
from src.skills.builtin.code_exec import CodeExecSkill, ShellExecSkill, WebFetchSkill


def register_builtin_skills():
    registry = get_registry()
    for skill_cls in [
        CalculatorSkill,
        FileOpsSkill,
        DateTimeSkill,
        SystemInfoSkill,
        CodeExecSkill,
        ShellExecSkill,
        WebFetchSkill,
    ]:
        instance = skill_cls()
        registry.register(instance)
