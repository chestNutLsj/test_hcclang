# Solver Module Tests

本目录包含对 `hcclang.solver` 模块的测试用例。

## 测试内容

### test_instance.py
测试 `Instance` 类的功能，包括：
- 基本对象创建和参数设置
- `rounds()` 方法计算
- `set()` 方法的不可变更新
- 字符串表示
- 对象的不可变性验证

### test_isomorphisms.py
测试 `isomorphisms` 模块的功能，包括：
- `Permutation` 类的创建和字符串表示
- `find_isomorphisms()` 函数的同构查找
- 错误处理（不同大小拓扑、无效参数）
- 不同拓扑类型的同构测试

### test_integration.py
集成测试，验证：
- 所有solver模块能正常导入
- 基本功能组件的实例化
- 与其他模块（topologies、collectives）的集成

## 运行测试

### 运行所有测试
```bash
cd tests/solver_test
python run_tests.py
```

### 运行单个测试文件
```bash
cd tests/solver_test
python test_instance.py
python test_isomorphisms.py
python test_integration.py
```

### 使用conda环境运行
```bash
conda activate dev312
cd tests/solver_test
python run_tests.py
```

## 测试结果输出

测试结果将输出到控制台，并保存在当前目录下。成功的测试会显示详细的运行信息，失败的测试会显示错误详情。

## 注意事项

1. 确保在运行测试前已经正确设置了Python路径
2. 测试依赖 z3 求解器，确保已正确安装
3. 部分测试可能需要较长时间完成，请耐心等待
4. 如果测试失败，请检查依赖模块是否正确安装

## 扩展测试

如需添加新的测试用例：
1. 在相应的测试文件中添加新的测试方法
2. 或创建新的 `test_*.py` 文件
3. 确保新测试遵循unittest框架的命名约定
4. 运行 `run_tests.py` 验证新测试正常工作 