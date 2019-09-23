import re
from collections import namedtuple, ChainMap
import os

from termcolor import cprint

os.system('color')

REGEXS = [(re.compile(i),j) for i,j in (
	(r'\s+','WHITESPACE'),
	(r'//[^\n]*|/\*(?s:.)*\*/','COMMENTS'),
	(r'if|while|else|var|def|return|bf|import','KEYWORD'),

	(r'\{\{[^}]*\}\}','BRAINFUCK_CODE'),
	(r'>=|<=|==|>|<|!=','COMPAR_OP'),
	(r'\+=|%=|-=|\*=|/=|=','ASSIGNEMENT_OP'),
	(r'\+\+|--','INC_DEC_OP'),
	(r'\+|\*\*|\*|-|/|%','BIN_OP'),
	(r'\(|\)|\{|\}|;|,','CONTROL_CHAR'),

	(r'[a-zA-Z]\w*','IDENTIFIERS'),
	(r'\d[\d_]*','NUMBERS'),
	(r"'.'",'CHAR'),
	(r'"(\\.|[^"\n])*"','STRING'),
)]

USELESS_TAGS = ('WHITESPACE','COMMENTS')

Token = namedtuple('Token',('text','tag'))

def lex(program,keep_whitespace=False):
	pos = 0
	tokens = []
	while pos < len(program):
		for regex, tag in REGEXS:
			match = regex.match(program,pos)
			if match is not None:
				token = match.group(0)
				tokens.append(Token(token,tag))
				pos = match.end(0)
				break
		else:
			raise SyntaxError(f"No match find during the lexing for {program[pos:pos+20]}...")
	if keep_whitespace:
		return tokens
	return clean_tokens(tokens)

def clean_tokens(tokens):
	return [token for token in tokens if token.tag not in USELESS_TAGS]

def format_bf(prog,debug_mode=True):
	if debug_mode:
		t = []
		index = 0
		while index < len(prog):
			if prog[index] in '[]<>+-.,':
				t.append(prog[index])
				index += 1
			elif prog[index] == '#':
				t.append('#')
				index += 1
				try:
					if prog[index] == '(':
						index += 1
						t.append('(')
						while prog[index] != ')':
							t.append(prog[index])
							index+=1
						t.append(')')
				except IndexError:
					pass
			else:
				index+=1
		return ''.join(t)

	return ''.join(i for i in prog if i in '[]<>+-.,')



#****************************


ProgramNode = namedtuple('ProgramNode',('list_of_statement','name'))
BlockNode = namedtuple('BlockNode',('list_of_statement'))

IfNode = namedtuple('IfNode',('condition','block','else_block'))
WhileNode = namedtuple('WhileNode',('condition','block'))
FunctionDefNode = namedtuple('FunctionDefNode',('name','args','block'))
BfDefNode = namedtuple('BfDefNode',('name','code'))

AssignementNode = namedtuple('AssignementNode',('target','operator','expression'))
IncDecNode = namedtuple('IncDecNode',('name','operator'))
DeclarationNode = namedtuple('DeclarationNode',('names'))
FunctionCallNode = namedtuple('FunctionCallNode',('name','args'))
ReturnNode = namedtuple('ReturnNode',('expr'))
ImportNode = namedtuple('ImportNode',('path',))
BinOpNode = namedtuple('BinOpNode',('left','op','right'))

IdentifierNode = namedtuple('IdentifierNode',('name'))
NumberNode = namedtuple('NumberNode',('value'))


class Parser:
	def __init__(self,tokens):
		self.tokens = tokens
		self.pos = 0

	def parse(self):
		return self.program()

	def accept_text(self,value):
		try:
			token = self.tokens[self.pos]
		except IndexError:
			#print("reach the end")
			return False
		if token.text == value:
			self.pos += 1
			return token
		return False

	def accept_tag(self,tag):
		try:
			token = self.tokens[self.pos]
		except IndexError:
			return False
		if token.tag == tag:
			self.pos += 1
			return token
		return False




	def program(self):
		nodes = []
		while 1:
			statement = self.statement()
			if isinstance(statement,ImportNode):
				ast = get_ast(statement.path)
				nodes.extend(ast)
			if statement:
				nodes.append(statement)
			else:
				break
		if not len(nodes):
			raise ValueError("Your program seems to be empty")
		return ProgramNode(nodes,'__main__') #plus simple pour le debug

	def block(self):
		#print(f"enter block ({self.pos})")
		if not self.accept_text("{"):
			return None
		nodes = []
		while 1:
			statement = self.statement()
			if statement:
				nodes.append(statement)
			else:
				break
		if not self.accept_text('}'):
			raise SyntaxError("'}' missing at the end of the block")

		#cprint("block",'green')
		return BlockNode(nodes)

	def statement(self):
		for func in (self.short_statement,self.while_block, self.if_block,self.function_block, self.bf_block):
			node = func()
			if node:
				return node
		return None

	def short_statement(self):
		#print(f"enter short_statement ({self.pos})")
		for func in (self.declaration, self.assignement,self.inc_dec,self.expression,self.return_,self.import_):
			node = func()
			if node:
				if self.accept_text(";"):
					#cprint('short_statement','green')
					return node
				else:
					raise SyntaxError(f"Missing ';' ({node})")
		#cprint("no short_statement",'red')
		return None

	def import_(self):
		if not self.accept_text('import'):
			return None
		path = self.accept_tag('STRING')
		if not path:
			raise SyntaxError('Bad import')
		return ImportNode(path.text[1:-1])

	def return_(self):
		if not self.accept_text('return'):
			return None
		expr = self.expression()
		if not expr:
			raise SyntaxError("Bad Return")
		return ReturnNode(expr)

	def declaration(self):
		if not self.accept_text('var'):
			return None
		name = self.accept_tag('IDENTIFIERS')
		if not name:
			raise SyntaxError("You must specifies an identifiers after a 'var' keyword")
		names = [IdentifierNode(name.text)]
		while 1:
			if self.accept_text(','):
				name = self.accept_tag('IDENTIFIERS')
				if name:
					names.append(IdentifierNode(name.text))
				else:
					raise SyntaxError("Expected an identifiers after a ',' in a declaration statement")
			else:
				break

		return DeclarationNode(names)

	def assignement(self):
		#print(f"enter assignement ({self.pos})")
		name = self.accept_tag('IDENTIFIERS')
		if not name:
			#cprint("no assignement (no name)",'red')
			return None
		sym = self.accept_tag('ASSIGNEMENT_OP')
		if not sym:
			self.pos -= 1 #for the name
			return None
		expr = self.expression()
		if not expr:
			raise SyntaxError("Missing expression in a assignement")
		#cprint("assignement",'green')
		return AssignementNode(IdentifierNode(name.text),sym.text,expr)

	def inc_dec(self):
		#print(f"enter inc_dec ({self.pos})")
		name = self.accept_tag('IDENTIFIERS')
		if not name:
			#cprint("no inc_dec (no name)",'red')
			return None
		op = self.accept_tag('INC_DEC_OP')
		if not op:
			self.pos -= 1 	#for the name
			#cprint("no inc_dec (no op)",'red')
			return None
		#cprint("inc_dec",'green')
		return IncDecNode(IdentifierNode(name.text),op.text)

	def expression(self):
		compar = self.compar()
		if compar:
			return compar

		return None

	def call_function(self):
		name = self.accept_tag('IDENTIFIERS')
		if not name:
			return None
		if not self.accept_text('('):
			self.pos -= 1
			return None

		expr = self.expression()
		args = []
		if expr:
			args.append(expr)
			while self.accept_text(','):
				expr = self.expression()
				if not expr:
					raise SyntaxError("missing argument")
				args.append(expr)
		if not self.accept_text(')'):
			raise SyntaxError("A function call must have a ')' at the end")

		return FunctionCallNode(name.text,args)


	def compar(self):
		calcul1 = self.calcul()
		if not calcul1:
			return None

		op = self.accept_tag('COMPAR_OP')
		if op:
			calcul2 = self.calcul()
			if not calcul2:
				raise SyntaxError("missing second term in calcul")
			return BinOpNode(calcul1,op.text,calcul2)
		return calcul1

	def calcul(self):
		#print(f"enter calcul ({self.pos})")
		"""There is calcul, terme, factor and val for differentiate between +/-, * or /, and **"""
		terme1 = self.terme()
		if not terme1:
			return None

		while 1:
			op = None
			sym = self.accept_text('+')
			if sym:
				op = sym
			sym = self.accept_text('-')
			if sym:
				op = sym
			if not op:
				return terme1

			if op:
				terme2 = self.terme()
				if not terme2:
					raise SyntaxError("missing second term in calcul")
				terme1 = BinOpNode(terme1,op.text,terme2)
		return terme1


	def terme(self):
		#print(f"enter terme ({self.pos})")
		factor1 = self.factor()
		if not factor1:
			return None

		while 1:
			op = None
			sym = self.accept_text('*')
			if sym:
				op = sym
			sym = self.accept_text('/')
			if sym:
				op = sym
			sym = self.accept_text('%')
			if sym:
				op = sym

			if op:
				factor2 = self.factor()
				if not factor2:
					raise SyntaxError("missing second factor in terme")
				factor1 = BinOpNode(factor1,op.text,factor2)
			else:
				break
		return factor1

	def factor(self):
		#print(f"enter factor ({self.pos})")
		val = self.value()
		if not val:
			return None
		while self.accept_text('**'):
				val2 = self.value()
				if not val2:
					raise SyntaxError("missing second val un factor")
				val =  BinOpNode(val,'**',val2)
		return val

	def value(self):
		#print(f"enter value ({self.pos})")	
		func = self.call_function()		#The orders matters !!!
		if func:
			return func

		name = self.accept_tag('IDENTIFIERS')
		if name:
			return IdentifierNode(name.text)
		
		number = self.accept_tag('NUMBERS')
		if number:
			return NumberNode(int(number.text))

		char = self.accept_tag('CHAR')
		if char:
			return NumberNode(ord(char.text[1]))

		if self.accept_text('('):
			expr = self.expression()
			if not (expr and self.accept_text(')')):
				raise SyntaxError("missing expression or matching ')'")
			return expr
		return None

	def while_block(self):
		#print(f"enter while_block ({self.pos})")		
		if not self.accept_text('while'):
			#cprint("no while_block (keyword)",'red')
			return None

		if not self.accept_text('('):
			raise SyntaxError("Bad while block")

		cond = self.expression()
		if not cond:
			raise SyntaxError("Bad while block")
		if not self.accept_text(')'):
			raise SyntaxError("Bad while block")
		block = self.block()
		if not block:
			raise SyntaxError("Bad while block")
		#cprint("while_block",'green')
		return WhileNode(cond,block)

	def if_block(self):
		#print(f"enter if_block ({self.pos})")
		if not self.accept_text('if'):
			#cprint("no if_block (keyword)",'red')
			return None

		if not self.accept_text('('):
			raise SyntaxError("Bad if block")

		cond = self.expression()
		if not cond:
			raise SyntaxError("Bad if block")
		if not self.accept_text(')'):
			raise SyntaxError("Bad if block")
		block = self.block()
		if not block:
			raise SyntaxError("Bad if block")

		if self.accept_text('else'):
			else_block = self.block()
			if not else_block:
				raise SyntaxError("Bad else block")
			return IfNode(cond,block,else_block)
		#cprint("if_block",'green')
		return IfNode(cond,block,None)

	def function_block(self):
		if not self.accept_text('def'):
			return None
		name = self.accept_tag('IDENTIFIERS')
		if not name:
			raise SyntaxError("A function need a name")
		if not self.accept_text('('):
			raise SyntaxError("Bad function block")
		arg = self.accept_tag('IDENTIFIERS')
		args = []
		if arg:
			args.append(IdentifierNode(arg.text))
			while self.accept_text(','):
				arg = self.accept_tag('IDENTIFIERS')
				if not arg:
					raise SyntaxError("Bad function block")
				args.append(IdentifierNode(arg.text))

		if not self.accept_text(')'):
			raise SyntaxError("Bad function block")

		block = self.block()
		if not block:
			raise SyntaxError("Bad function block")

		return FunctionDefNode(name.text,args,block)

	def bf_block(self):
		if not self.accept_text('bf'):
			return None
		name = self.accept_tag('IDENTIFIERS')
		if not name:
			raise SyntaxError("A bf function need a name")

		code = self.accept_tag('BRAINFUCK_CODE')
		if not code:
			raise SyntaxError("Bad bf block")
		code = format_bf(code.text[2:-2])
		return BfDefNode(name.text,code)




def parse(prog):
	tokens = clean_tokens(lex(prog))
	parser = Parser(tokens)
	return parser.program()


def get_ast(path):
	with open(path) as file:
		code = file.read()
	tokens = lex(code)
	ast = Parser(tokens).parse()
	return ast.list_of_statement



#***************


class Ast_to_IR:

	def __init__(self,ast,functions=None):
		if functions is None:
			functions = ChainMap({})
		self.ast = ast
		self.functions = functions

	def add(self,val):
		self.ir.append(val)

	def convert(self):

		if isinstance(self.ast,ProgramNode):
			statements = self.ast.list_of_statement
			args = []
			block = self.ast
		elif isinstance(self.ast,FunctionDefNode):
			statements = self.ast.block.list_of_statement
			args = self.ast.args
			block = self.ast.block


		self.ir = []
		self.functions = self.functions.new_child()
		for statement in statements:
			if isinstance(statement,FunctionDefNode):
				self.functions[statement.name] = Ast_to_IR(statement,self.functions).convert()
			elif isinstance(statement,BfDefNode):
				code = statement.code
				self.functions[statement.name] = [('BF',code,statement.name)]


		vars_ = []
		for statement in statements:
			if isinstance(statement,DeclarationNode):
				vars_.extend(statement.names)

		all_vars = {val:index for index,val in enumerate((args+vars_)[::-1])}
		self.vars = all_vars
		self.pos = 0

		nb_decalage = len(vars_) - (1 if isinstance(self.ast,ProgramNode) else 0)
		self.add(('INIT_FUNCTION',nb_decalage ,self.ast.name))
		self.push_block(block)
		if isinstance(self.ast,FunctionDefNode):
			self.add(('END_FUNCTION',len(all_vars),self.ast.name))
		return self.ir

	def push_block(self,node):
		for statement in node.list_of_statement:
			func = methods.get(type(statement),default)	#switch-case
			func(self,statement)

	def assignement_to_ir(self,node):
		if node.operator == '=':
			expression = node.expression
		else:
			operator = node.operator[0]	#ex : '+=' -> '+'
			expression = BinOpNode(node.target,operator,node.expression)

		self.push_expr(expression)
		assign_distance = self.pos + self.vars[node.target]

		self.add(('ASSIGN',assign_distance,node.target.name))
		self.pos -= 1

	def return_to_ir(self,node):
		self.push_expr(node.expr)

	def expression_statement_to_ir(self,node):
		self.push_expr(node)
		self.pos -= 1
		self.add(('POP',))

	def inc_dec_to_ir(self,node):
		assign_distance = self.pos + self.vars[node.name]
		if node.operator == "++":
			self.add(('INC',assign_distance,node.name.name))
		else:
			self.add(('DEC',assign_distance,node.name.name))

	def push_expr(self,node):

		if isinstance(node,FunctionCallNode):
			for arg in node.args:
				self.push_expr(arg)

			try:
				self.ir.extend(self.functions[node.name])
			except KeyError:
				raise SyntaxError(f"The function {node.name} doesn't exist")
			self.pos -= len(node.args)
			self.pos += 1
			return

		if isinstance(node,NumberNode):
			self.pos += 1
			self.add(('PUSH_NUMBER',node.value))
			return
		if isinstance(node,IdentifierNode):
			self.pos += 1
			distance = self.pos + self.vars[node]
			self.add(('PUSH_VAR',distance,node.name))
			return

		if isinstance(node,BinOpNode):
			self.push_expr(node.left)
			self.push_expr(node.right)
			self.pos -= 1
			self.add(('BIN_OP',node.op))
			return

		raise Exception(f"cette expression n'est pas bonne : {node}")

	def while_to_ir(self,node):
		self.push_expr(node.condition)
		self.add(('WHILE_ENTER',))
		self.pos -= 1
		self.push_block(node.block)
		self.push_expr(node.condition)
		self.add(('WHILE_END',))
		self.pos -= 1

	def if_to_ir(self,node):
		self.push_expr(node.condition)
		self.add(('IF_ENTER',))
		self.pos += 1
		self.push_block(node.block)
		if node.else_block:
			self.add(('ELSE_ENTER',))
			self.pos -= 2
			self.push_block(node.else_block)
			self.add(('ELSE_END',))
		else:
			self.add(('IF_END',))
			self.pos -= 2



methods = {
	DeclarationNode: lambda self,node:None,
	FunctionDefNode: lambda self,node:None,
	AssignementNode: Ast_to_IR.assignement_to_ir,
	ReturnNode: Ast_to_IR.return_to_ir,
	WhileNode: Ast_to_IR.while_to_ir,
	IfNode: Ast_to_IR.if_to_ir,
	IncDecNode: Ast_to_IR.inc_dec_to_ir,
	BfDefNode: lambda self,node:None,
	ImportNode: lambda self,node:None,
}
default = Ast_to_IR.expression_statement_to_ir


#************



def init_function(nb,name):
	return '>' * nb + f'	INIT_FUNCTION {name}'

def end_function(nb,name):
	return '<[-]' * nb + '>' * nb + (f"[-{'<'*nb}+{'>'*nb}]" if nb else "") + '<' * nb+ f'	END_FUNCTION {name}'

def bf(code,name):
	return f"BF {name}\n{code}\nEND_BF"



def assign(val,name):
	return f"{'<'*val}[-]{'>'*val}[-{'<'*val}+{'>'*val}]<	ASSIGN {name}"

def inc(val,name):
	return f"{'<'*val}+{'>'*val}	INC {name}"

def dec(val,name):
	return f"{'<'*val}-{'>'*val}	DEC {name}"

def pop():
	return '[-]<	POP'

def push_number(val):
	return ">" + "+"*val + f"	PUSH_NUMBER {val}"

def push_var(val,name):
	return '<' * (val-1) + f"[-{'>'*val}+>+<{'<'*val}]" + f"{'>'*val}>[-<{'<'*val}+>{'>'*val}]<" + f"	PUSH_VAR {name}"

def bin_op(op):
	if op == '+':
		bf = "[-<+>]<"
	elif op == '-':
		bf =  "[-<->]<"
	elif op == '*':
		bf = "<[->>+<<]>[->[-<<+>>>+<]>[-<+>]<<]>[-]<<"
	elif op == '/':
		bf = ("[->+>>+<<<]<[->+>>+<<<]>>>[>[->+>+<<]>>[-<<+>>]+<[>-<[-]]>[->+>+<<]>>[-<<+>>]<[<<<<[-]+>+>>>-]<<<-<-]+>[<->[-]]>>[-]<<<"
			"[-<[-<->>+<]>[-<+>]<<<+>[->>+>+<<<]>>>[-<<<+>>>]<<[->>+>+<<<]>>>[-<<<+>>>]<<[>[->+>+<<]>>[-<<+>>]+<[>-<[-]]>[->+>+<<]>"
			">[-<<+>>]<[<<<<[-]+>+>>>-]<<<-<-]+>[<->[-]]>>[-]<<<]<[-]<[-]<")
	elif op == '%':
		bf = "[->+>>+<<<]<[->+>>+<<<]>>>[>[->+>+<<]>>[-<<+>>]+<[>-<[-]]>[->+>+<<]>>[-<<+>>]<[<<<<[-]+>+>>>-]<<<-<-]+>[<->[-]]>>[-]<<<" \
			"[-<[-<->>+<]>[-<+>]<<[->>+>+<<<]>>>[-<<<+>>>]<<[->>+>+<<<]>>>[-<<<+>>>]<<[>[->+>+<<]>>[-<<+>>]+<[>-<[-]]>[->+>+<<]>>" \
			"[-<<+>>]<[<<<<[-]+>+>>>-]<<<-<-]+>[<->[-]]>>[-]<<<]<[-]<[-<+>]<"
	elif op == '==':
		bf = "<[>[->+>+<<]>>[-<<+>>]+<[>-<[-]]>[->+>+<<]>>[-<<+>>]<[<<<<[-]+>+>>>-]<<<-<-]" + "+>[<->[-]]>>[<<<->>>-]<<<"
	elif op == '!=':
		bf = "<[>[->+>+<<]>>[-<<+>>]+<[>-<[-]]>[->+>+<<]>>[-<<+>>]<[<<<<[-]+>+>>>-]<<<-<-]" + ">[<+>[-]]>>[<<<+>>>-]<<<"
	elif op == '>':
		bf = "<[>[->+>+<<]>>[-<<+>>]+<[>-<[-]]>[->+>+<<]>>[-<<+>>]<[<<<<[-]+>+>>>-]<<<-<-]" + ">[-]>>[<<<+>>>-]<<<"
	elif op == '>=':
		bf = "<[>[->+>+<<]>>[-<<+>>]+<[>-<[-]]>[->+>+<<]>>[-<<+>>]<[<<<<[-]+>+>>>-]<<<-<-]" + "+>[<->[-]]>>[-]<<<"
	elif op == '<':
		bf = "<[>[->+>+<<]>>[-<<+>>]+<[>-<[-]]>[->+>+<<]>>[-<<+>>]<[<<<<[-]+>+>>>-]<<<-<-]" + ">[<+>[-]]>>[-]<<<"
	elif op == '<=':
		bf = "<[>[->+>+<<]>>[-<<+>>]+<[>-<[-]]>[->+>+<<]>>[-<<+>>]<[<<<<[-]+>+>>>-]<<<-<-]" + "+>[-]>>[<<<->>>-]<<<"

	else:
		raise Exception(f"bin_op {op} not implemented yet")
	return bf + f"	BIN_OP " +\
	(op if op not in ('+','-','<','>','<=','>=') else {'+':'plus','-':'minus','>':'gt','<':'st','>=':'gt_eq','<=':'st_eq'}[op])

def while_enter():
	return "[[-]<	WHILE_ENTER"

def while_end():
	return "]<	WHILE_END"

def if_enter():
	return ">+<[[-]>-	IF_ENTER"

def else_enter():
	return "<]>[-<<	ELSE_ENTER"

def else_end():
	return ">>]<<	ELSE_END"

def if_end():
	return "<]>[-]<<	IF_END"


def compile_ir(ir):
	return '\n'.join(globals()[i[0].lower()](*i[1:]) for i in ir)



#**************

def compile(prog,level=4):
	tokens = lex(prog)
	if level == 1:
		return tokens
	ast = Parser(tokens).parse()
	if level == 2:
		return ast
	ir = Ast_to_IR(ast).convert()
	if level == 3:
		return ir
	brainfuck = compile_ir(ir)
	return brainfuck


def main():
	import sys
	try:
		file = sys.argv[1]
	except IndexError:
		raise Exception("You must set a file")

	prog = open(file).read()
	bf_prog = compile(prog)
	with open(os.path.splitext(file)[0]+'.bf','w') as f:
		f.write(bf_prog)


if __name__ == '__main__':
	main()

