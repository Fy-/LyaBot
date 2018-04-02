
import os
import ast
import regex as re
from settings import settings

def prepare_default():
	src_lines = []
	tgt_lines = []
	with open(os.path.join(settings.path_static, 'default.src'), 'r',  encoding='utf-8') as src:
		with open(os.path.join(settings.path_static, 'default.tgt'), 'r',  encoding='utf-8') as tgt:
			for s, t in zip(src, tgt):
				src_lines.append(s.replace('\n', '').strip())
				tgt_lines.append(t.replace('\n', '').strip())

	return src_lines, tgt_lines

def prepare_cornerll():
	line_file = os.path.join(settings.path_static, 'cornell', "movie_lines.txt")
	conv_file = os.path.join(settings.path_static,  'cornell', "movie_conversations.txt")
	lines_fields =  ["lineID","characterID","movieID","character","text"]
	convs_fields =  ["character1ID","character2ID","movieID","utteranceIDs"]

	def load_conversations(filename, lines, fields):
		convs = []
		with open(filename, 'r', encoding='utf-8') as f:
			for line in f:
				values = line.split(" +++$+++ ")

				c = {}
				for i, field in enumerate(fields):
					c[field] = values[i]

				line_ids = ast.literal_eval(c["utteranceIDs"])

				c["lines"] = []
				for line_id in line_ids:
					c["lines"].append(lines[line_id])

				convs.append(c)

		return convs

	def load_lines(filename, fields):
		lines = {} 
		with open(filename, 'r', encoding='utf-8') as f:
			for line in f:
				values = line.split(" +++$+++ ")
				l = {}
				for i, field in enumerate(fields):
					l[field] = values[i]
				lines[l['lineID']] = l

		return lines



	lines = load_lines(line_file, lines_fields)
	convs = load_conversations(conv_file, lines, convs_fields)
	return convs


def main():
	src_lines, tgt_lines = prepare_default()
	cornell = prepare_cornerll()

	def prepare(html, dot, dash, spaces, sentence):
		sentence = spaces.sub(' ', sentence)


		sentence = dot.sub(', ', sentence)
		sentence = dash.sub(', ', sentence)
		sentence = sentence.replace(' \' ', '\'')
		sentence = re.sub(html, '', sentence)

		return sentence

	tokens = [
			'CALL_GET_NAME', 'CALL_GET_MYFULLNAME', 'CALL_GET_TIME', 'CALL_GET_TODAY_DAY', 
			'CALL_GET_TODAY_DATE', 'CALL_GET_YEAR', 'CALL_GET_YESTERDAY_DAY', 'CALL_GET_YESTERDAY_DATE', 
			'CALL_GET_TOMORROW_DAY', 'CALL_GET_TOMORROW_DATE', 'CALL_GET_MYLOCATION', 
			'CALL_GET_MYURL', 'CALL_MYAGE', 'CALL_MYUPTIME', 'CALL_GET_JOKE' 
		]
	def lower(sentence):
		sentence = sentence.lower()
		for token in tokens:
			sentence = sentence.replace(token.lower(), token)
		return sentence

	with open(os.path.join(settings.path_data, 'data.src'), 'w', encoding='utf-8') as src_f:
		with open(os.path.join(settings.path_data, 'data.tgt'), 'w', encoding='utf-8') as tgt_f:
			src_lines_lower = [lower(line) for line in src_lines]
			tgt_lines_lower = [lower(line) for line in tgt_lines]

			src_lines = '\n'.join(src_lines)+'\n'
			tgt_lines = '\n'.join(tgt_lines)+'\n'
			src_lines_low = '\n'.join(src_lines_lower)+'\n'
			tgt_lines_low = '\n'.join(tgt_lines_lower)+'\n'

			for i in range(100):
				src_f.write(src_lines)
				tgt_f.write(tgt_lines)
			for i in range(100):
				src_f.write(src_lines_low)
				tgt_f.write(tgt_lines_low)

			html = re.compile(r'<.*?>')
			dot =  re.compile(r' \.{2,} ')
			dash = re.compile(r'\-{2,}')
			spaces = re.compile(r'[^\S\n]+')
			ban = re.compile(r'(asshole|bitch|child-fucker|motherfucker|motherfucking|nigger|bitch)', re.IGNORECASE)

			print (cornell[0])

			for c in cornell:
				for i in range(0, len(c['lines']) - 1, 2):
					src_line = c['lines'][i]['text'].strip()
					target_line = c['lines'][i + 1]['text'].strip()
					
					if src_line.startswith("...") or target_line.startswith("..."):
						continue

					if src_line.startswith("-") or target_line.startswith("-"):
						continue


					if src_line.endswith("--") or target_line.endswith("--"):
						continue

					if len(src_line.split()) < 3 or len(src_line.split()) > 50 or len(target_line.split()) < 3 or len(target_line.split()) > 50:
						continue

					if ban.findall(target_line):
						continue

					src_f.write(prepare(html, dot, dash, spaces, src_line) + '\n')
					tgt_f.write(prepare(html, dot, dash, spaces, target_line) + '\n')

if __name__ == "__main__":
	main()