import chex
import jax.numpy as jnp

def run_simple_test():
    # 模拟基础维度
    B = 256  # Batch Size
    D = 10   # Action Dim
    
    # 构造测试变量
    a = jnp.ones((B, D))
    b = jnp.ones((B, D))  # a 和 b 形状完全一样
    c = jnp.ones((B,))     # c 只是一个一维向量
    
    # 构造一个树结构（字典），模拟 SAC 中的 batch
    # 里面所有的叶子节点第一个维度都是 B
    my_tree = {
        "obs": jnp.ones((B, 3, 64, 64)),
        "act": jnp.ones((B, D)),
        "rew": jnp.ones((B,))
    }

    print("--- 1. 测试 assert_equal_shape (横向对比) ---")
    # 只要 a 和 b 的形状一模一样就通过
    chex.assert_equal_shape([a, b])
    print("✅ a 和 b 形状对齐了！")

    print("\n--- 2. 测试 assert_shape (纵向核对) ---")
    # c 的形状必须死死地等于 (256,)
    chex.assert_shape(c, (B,))
    print("✅ c 的规格完全正确！")

    print("\n--- 3. 测试 assert_tree_shape_prefix (批量前缀核对) ---")
    # 检查 my_tree 字典里每一个数组的开头是不是 (256,)
    # 它不管 obs 是四维，act 是二维，rew 是一维，只要它们第一维都是 256 就行
    chex.assert_tree_shape_prefix(my_tree, (B,))
    print("✅ 字典里所有数据的 Batch Size 都对上了！")

    # ==========================================
    # 故意触发报错，看看 Chex 怎么说
    # ==========================================
    print("\n--- 4. 模拟报错：当形状不匹配时 ---")
    try:
        # 尝试让 a (256, 10) 和 c (256,) 比形状
        chex.assert_equal_shape([a, c])
    except AssertionError as e:
        print(f"❌ 报错成功！信息如下：\n{e}")

    print("\n--- 5. 模拟报错：当树的前缀不一致时 ---")
    bad_tree = {
        "obs": jnp.ones((B, 3)),
        "act": jnp.ones((128, D)) # 这里的 Batch 只有 128，掉队了
    }
    try:
        chex.assert_tree_shape_prefix(bad_tree, (B,))
    except AssertionError as e:
        print(f"❌ 树结构检测失败！报错：\n{e}")

if __name__ == "__main__":
    run_simple_test()