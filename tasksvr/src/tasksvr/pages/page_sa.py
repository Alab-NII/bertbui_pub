# coding: utf-8


import json
import tornado.web
from .page_base import (
    TaskHandlerBase, TextProcessMixin, 
    SCRIPT_BASE_HEAD, SCRIPT_BASE_TAIL, STYLE_BASE,
    STYLE_TEXTBOX, STYLE_CHECKBOX
)


class TSearchAndAnswerInnerPage(tornado.web.RequestHandler, TextProcessMixin):
    
    def initialize(self, datasets):
        
        self.datasets = datasets
        self.is_debugging = False
    
    @staticmethod
    def search(example, query):
        query = query.lower()
        results = []
        for question in example['questions']:
            if query in question['index']:
                results.append(question)
        return results
    
    def get(self):
        
        sid = self.get_argument('sid', None) # split
        eid = self.get_argument('eid', None) # example
        query = self.get_argument('q', None)
        detail_qid = self.get_argument('qid', None)
        
        src = '<!DOCTYPE html>'
        src += '<html style="overflow-y: scroll;">'
        src += '<head>'
        src += '<script>'+SCRIPT_BASE_HEAD+'</script>'
        src += '<style>'
        src += STYLE_BASE
        src += STYLE_TEXTBOX
        src += 'table, th, td {border:1px solid black;}'
        src += 'th, td {padding-left: 5px; padding-right: 5px;}'
        src += '</style>'
        src += '</head>'
        
        src += '<body>'
        
        if detail_qid is not None:
            example = self.datasets[str(sid)][int(eid)]
            question = example['questions'][int(detail_qid)]
            qid = question['qid']
            question_str = str(question['question'])
            cid = question['cid']
            context = example['contexts'][cid]
            ctype = context['context_type']
            if ctype == 'text':
                content = context['text']
            else:
                image_path = context['image_path']
                style = 'display: block; margin-left: auto; margin-right: auto;'
                content = f'<img src="{image_path}" width="80%" style="{style}"/>'
            src += '<table style="border-collapse: collapse;">'
            src += f'<tr><td>QID</td><td>{qid}</td></tr>'
            src += f'<tr><td>Question</td><td>{question_str}</td></tr>'
            src += f'<tr><td>CID</td><td>{cid}</td></tr>'
            src += f'<tr><td>Context</td><td>{content}</td></div>'
            src += '</table>'
            
        else:
            # write search window
            src += f'<form action="" autocomplete="off" style="display:inline-block;">'
            src += f'<input type="hidden" name="sid" value="{sid}"/><input type="hidden" name="eid" value="{eid}"/>'
            value = query if query else ''
            src += f'<input type="text" id="input_query" name="q" style="width: 300px;" value="{value}" /> '
            src += '<button id="submit" style="padding: 4px 8px;">search</button>'
            src += '</form>'
        
            if query is not None:
                # write search results
                example = self.datasets[str(sid)][int(eid)]
                query = str(query)
                results = self.search(example, query)
                src += '<hr/>'
                src += 'hit: %d<br/>' % len(results)
                # table
                src += '<table style="border-collapse: collapse;">'
                src += '<tr><th>QID</th><th>CID</th><th>Type</th><th>Question</th><th>Detail</th></tr>'
                for result in results:
                    qid = result['qid']
                    qid_int = int(result['qid'].strip('Q'))
                    cid = result['cid']
                    ctype = example['contexts'][cid]['context_type']
                    question = result['question']
                    detail = f'<a href="?sid={sid}&eid={eid}&qid={qid_int}" id="show_{qid}" style="color: blue;">show</a>'
                    src += f'<tr><td id="qid_{qid}">{qid}</td><td>{cid}</td><td>{ctype}</td><td>{question}</td><td>{detail}</td></tr>'
                src += '</table>'
        
        src += '<script>'+SCRIPT_BASE_TAIL+'</script>'
        src += '</body></html>'
        src = self.emulated_ocr(src, self.is_debugging)
        
        self.write(src)
    
    
class TSearchAndAnswer(TaskHandlerBase, TextProcessMixin):
    
    def get_html_text(self, _id):
        
        fields = self.compile_fields(_id)
        text = self.render_text(**fields)
        return text    
    
    def compile_fields(self, _id):
        
        data = self.data[_id]
        metadata = self.metadata
        fields = {}
        
        fields['query'] = data['gold'][0][1]
        
        gold_actions = 'iframe_move_to+frame+#input_query click key_stroke+query '+ \
            'iframe_move_to+frame+#submit click '
        
        if len(data['gold']) > 1:
            if data['gold'][1][0] == 'click':
                show_id = data['gold'][1][1]
                gold_actions  += f'iframe_move_to+frame+{show_id} click iframe_scroll_down+frame '
            elif data['gold'][1][0] == 'move':
                qid = data['gold'][1][1]
                gold_actions  += f'iframe_move_to+frame+{qid} '

        if data['answer']:
            gold_actions += 'move_to+#input_answer click key_stroke+answer move_to+#submit click'
        else:
            gold_actions += 'move_to+#cb_unanswerable click move_to+#submit click'
        
        fields['gold_actions'] = gold_actions
        fields['instruction'] = 'Search and Answer Task'
        fields['home_url'] = data['home_url']
        fields['question'] = data['question']
        fields['answer'] = data['answer']
        
        return fields

    def render_text(self, instruction, query, home_url, question, answer, gold_actions):
            
        font_size_head, font_size_main, font_size_tail, line_height = ('16px', '16px', '16px', 1.4)
            
        # Escaping
        # we do not need escape except for newline for target sentences of emulated ocr.
        instruction = self.escape_new_line(instruction)
        question = self.escape_new_line(question)
        
        style = 'padding: 5px;'
        context_q = f'<div style="{style}"><strong>Question:</strong> {question}<br/></div>'
        
        src = '<!DOCTYPE html>'
        src += '<html style="overflow-y: scroll;">'
        
        src += '<head>'
        src += '<meta name="status" content=""><meta name="submitted" content="">'
        src += f'<meta name="teacher_actions" content="{gold_actions}">'
        src += f'<meta name="answer" content="{self.escape_double_quote(answer)}">'
        src += f'<meta name="query" content="{self.escape_double_quote(query)}">'
        src += '<style>'
        src += STYLE_BASE
        src += STYLE_TEXTBOX
        src += STYLE_CHECKBOX
        src += '</style>'
        src += '<script>' + SCRIPT_BASE_HEAD + '</script>'
        src += '</head>'
        
        src += '<body>'
        style = f'display:block; margin-bottom:10px; font-size:{font_size_head}; line-height:{line_height};'
        src += f'<div class="div_head" style="{style}">'
        src += f'{instruction}<br/>{context_q}'
        src += '</div>'
            
        style = f'display:block; margin-bottom:10px; font-size:{font_size_main}; line-height:{line_height};'
        src += f'<div class="div_main" style="{style}">'
        src += '<div style="display:inline; padding-right: 20px">QA Database Search</div>'
        onclick = f'document.getElementById(\'frame\').src=\'{home_url}\''
        src += f'<button id="home" style="padding: 4px 8px;" onclick="{onclick}">Home</button>'
        src += f'<button id="prev" style="margin-left: 5px; padding: 4px 8px;" onclick="history.go(-1)"><< Prev</button>'
        src += f'<button id="next" style="margin-left: 5px; padding: 4px 8px;" onclick="history.go(1)">Next >></button><br/>'
        src += '<iframe id="frame" style="border-style: solid; border-width: 1px; display: block; margin-left: auto; margin-right: auto;"></iframe>'
        src += '</div>'
        
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_tail}; line-height:{line_height};'
        src += f'<div class="div_tail" style="{style}">'
        src += '<div style="display: inline; margin-bottom: 10px;">Answer: </div>'
        src += '<form action="javascript:void(0);" style="display:inline;" autocomplete="off">'
        src += '<input type="text" id="input_answer" style="width: 200px;" /> '
        
        src += '<label class="mycheckbox">'
        src += '<input type="checkbox" value="unanswerable" id="unanswerable" />'
        src += '<span class="checkmark"  id="cb_unanswerable"></span> unanswerable '
        src += '</label>'
        
        onclick = 'task_submit({unanswerable:document.getElementById(\'unanswerable\').checked, ' + \
                    'answer:document.getElementById(\'input_answer\').value})'
        src += f'<button id="submit" style="vertical-align: bottom; padding: 4px 8px;" onclick="{onclick}">submit</button>'
        src += '</form>'
        src += '</div>'
        
        # Fit iframe size
        src += '<script>'
        src += 'var frame = document.getElementById("frame");'
        src += 'frame.width = 0.95*window.innerWidth +"px";'
        src += 'frame.height = 0.6*window.innerHeight +"px";'
        src += f'frame.src = "{home_url}";'
        src += SCRIPT_BASE_TAIL
        src += '</script>'
        
        src += '</body></html>'
            
        src = self.emulated_ocr(src, self.is_debugging)
        return src
