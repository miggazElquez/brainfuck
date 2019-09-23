from collections import ChainMap

import to_bf_compiler as compiler

BUILT_IN_FUNCTIONS = ChainMap({})

BUILT_IN_FUNCTIONS['putc'] = [('PUTC',)]
BUILT_IN_FUNCTIONS['scanc'] =[('SCANC',)]
BUILT_IN_FUNCTIONS['debug'] = [('DEBUG',)]

func_to_compile = []



func_to_compile.append(
	('print_int',"""

def print_int(n){
	var i, val;
	i = 1;
	while (i <= n){
		i *= 10;
	}
	i /= 10;
	while (i >= 1){
		val = n / i;
		n %= i;
		putc(val + 48);
		i /= 10;
	}
	return n;
}
"""))



def compile_func(name,func):
	ast = compiler.compile(func,2)
	ir = compiler.Ast_to_IR(ast,BUILT_IN_FUNCTIONS) #on peut utiliser toutes les fonctions définis avant
	ir.convert()
	return ir.functions[name]



for name, func in func_to_compile:
	BUILT_IN_FUNCTIONS[name] = compile_func(name,func)


