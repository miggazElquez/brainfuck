"""interpreter of brainfuck written in python"""
import sys
import os

from termcolor import cprint

os.system('color')

class BFError(RuntimeError):
	pass

def pair_bracket(prog):
	result = []
	brackets = []
	for index, instr in enumerate(prog):
		if instr == '[':
			brackets.append(index)
		if instr == ']':
			open_brackets = brackets.pop()
			result.append((open_brackets,index))
	return dict(result)


def interpret(prog):
	ar = [0]
	ptr = 0
	index = 0
	blocks = []
	bracket_info = pair_bracket(prog)
	while index < len(prog):
		instr = prog[index]

		if instr == '+':
			ar[ptr] += 1
		elif instr == '-':
			ar[ptr] -= 1
		elif instr == '>':
			ptr += 1
			if ptr == len(ar):
				ar.append(0)
		elif instr == '<':
			ptr -= 1
			if ptr < 0:
				raise ValueError("You can't go to a cell before 0")
		elif instr == '.':
			print(chr(ar[ptr]),end='')
		elif instr == ',':
			val = input()
			if len(val) > 1:
				raise ValueError('please put character one by one')
			if val:
				ar[ptr] = ord(val)
			else:
				ar[ptr] = 0
		elif instr == '[':
			if ar[ptr]:
				blocks.append(index)
			else:
				index = bracket_info[index]
		elif instr == ']':
			if ar[ptr]:
				index = blocks.pop()
				continue
		elif instr == '#':
			print(ar, ptr, ar[ptr])

		index += 1

	return ar


def format(prog,debug_mode=True):
	if debug_mode:
		return ''.join(i for i in prog if i in '[]<>+-.,#')
	return ''.join(i for i in prog if i in '[]<>+-.,')


def repl():
	"""
	use brainfuck in an interactive mode

	special commands :
		p: where is the pointer
		v: val of the cell (the number, not the ascii val)
		a: show the entire array

	"""
	ar = [0]
	ptr = 0
	blocks = []
	while 1:
		try:
			not_ok = True
			while not_ok:
				try:
					prog = input('> ')
					not_ok = False
				except KeyboardInterrupt:
					print()
		except EOFError:
			break

		if prog == 'p':
			print(ptr)
		elif prog == 'v':
			print(ar[ptr])
		elif prog == 'a':
			print(ar)
		elif prog == 'help':
			repl_help()
		elif prog == 'q':
			break
		elif prog == 'r':
			repl()
			return
		else:
			index = 0
			bracket_info = pair_bracket(prog)
			try:
				while index < len(prog):
					instr = prog[index]

					if instr == '+':
						ar[ptr] += 1
					elif instr == '-':
						ar[ptr] -= 1
					elif instr == '>':
						ptr += 1
						if ptr == len(ar):
							ar.append(0)
					elif instr == '<':
						ptr -= 1
						if ptr < 0:
							ptr = 0
							raise BFError(f"You can't go to a cell before 0 (code before : {prog[:index]}, code after : {prog[index:]})")
					elif instr == '.':
						print(chr(ar[ptr]),end='')
					elif instr == ',':
						val = input()
						if len(val) > 1:
							raise ValueError('please put character one by one')
						if val:
							ar[ptr] = ord(val)
						else:
							ar[ptr] = 0
					elif instr == '[':
						if ar[ptr]:
							blocks.append(index)
						else:
							index = bracket_info[index]
					elif instr == ']':
						if ar[ptr]:
							index = blocks.pop()
							continue
						else:
							blocks.pop()
					elif instr == '#':
						try:
							if prog[index+1] == '(':
								index = index+2
								while prog[index] != ')':
									print(prog[index],end='')
									index += 1
								print(' ',end='')
						except IndexError:
							pass 
						print(ar, ptr, ar[ptr])

					index += 1
			except KeyboardInterrupt:
				pass
			except BFError as e:
				cprint(e,'red')

			if '.' in prog:
				print()


def repl_help():
	print("""\
In this repl, you can enter any valid brainfuck expression.
There is also some special command : 
	'a' will show you the state of the memory
	'p' will show you where is the pointer
	'v' will show you the value of the cell
	'r' will reset the array""")



def main():
	if len(sys.argv) == 2:
		interpret(open(sys.argv[2]).read())
	elif len(sys.argv) == 3:
		if sys.argv[1] == '-i':		#oui c'est de lam erde comme manière de faire
			interpret(sys.argv[2])
	else:
		repl()


if __name__ == '__main__':
	main()
