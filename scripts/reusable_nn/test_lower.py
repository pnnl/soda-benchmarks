# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Regression checks for semantic rewrites and residual buffer lifetimes."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from lower import Node, transpose, pointwise, fuse_relu, plan_memory, invocation


class LoweringTests(unittest.TestCase):
    def test_layout_propagation(self):
        a, b = Node("input", (1, 3, 5, 7)), Node("input", (1, 3, 5, 7))
        p, inverse = (0, 2, 3, 1), (0, 3, 1, 2)
        result = transpose(pointwise("add", (transpose(a,p),transpose(b,p)), {}), inverse)
        self.assertEqual(result.kind, "add")
        self.assertEqual(result.inputs, (a,b))
        self.assertIs(transpose(transpose(a,p),inverse), a)

    def test_constant_layout(self):
        data=np.arange(24,dtype=np.float32).reshape(2,3,4)
        result=transpose(Node("const",data.shape,data=data),(2,0,1))
        np.testing.assert_array_equal(result.data,data.transpose(2,0,1))

    def test_relu_does_not_change_another_consumer(self):
        x=Node("input",(1,2,2,3))
        conv=Node("conv2d",x.shape,(x,))
        relu=Node("relu",x.shape,(conv,))
        result=Node("add",x.shape,(relu,conv))
        with self.assertRaisesRegex(ValueError,"single-use"):
            fuse_relu(result)
        self.assertNotIn("relu",conv.attrs)

    def test_relu_single_use(self):
        source=Node("add",(5,), (Node("input",(5,)),Node("input",(5,))))
        result,count=fuse_relu(Node("relu",(5,),(source,)))
        self.assertIs(result,source)
        self.assertEqual(count,1)
        self.assertEqual(result.attrs["relu"],1)

    def test_residual_and_reshape_lifetimes(self):
        x=Node("input",(16,))
        weights=Node("const",(16,),data=np.ones(16,dtype=np.float32))
        a=Node("add",(16,),(x,weights))
        alias=Node("reshape",(4,4),(a,))
        b=Node("add",(16,),(a,weights))
        c=Node("add",(16,),(b,weights))
        result=Node("add",(4,4),(alias,Node("reshape",(4,4),(c,))))
        with tempfile.TemporaryDirectory() as directory:
            nodes,calls,parameters,arena=plan_memory(x,result,Path(directory))
        self.assertEqual(parameters,16)
        self.assertEqual(alias.offset,a.offset)
        # The skip tensor remains intact while the main branch advances.
        for node in (b,c,result):
            self.assertNotEqual(node.offset,a.offset)
        self.assertLess(arena,sum(n.elements*8 for n in nodes if n.kind!="reshape"))
        for node in calls:
            for parent in node.inputs:
                self.assertTrue(node.offset+node.elements*8<=parent.offset or
                                parent.offset+parent.elements*8<=node.offset)

    def test_invalid_gemm(self):
        a,b=Node("input",(1,2,3)),Node("const",(1,4,5))
        with self.assertRaisesRegex(ValueError,"Incompatible GEMM"):
            invocation(Node("matmul",(1,2,5),(a,b)))

    def test_unsupported_broadcast(self):
        with self.assertRaisesRegex(ValueError,"equal shapes"):
            pointwise("add",(Node("input",(1,5)),Node("const",(5,))),{})


if __name__=="__main__":
    unittest.main()
