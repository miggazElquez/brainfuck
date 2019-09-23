/*
STDLIB
The comment after the bf function is the number of arguments
*/


bf putc {{	//1
	.
}}

bf scanc {{	//0
	>,
}}

bf debug {{	//0
	#>
}}

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
