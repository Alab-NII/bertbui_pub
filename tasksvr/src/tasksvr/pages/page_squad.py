# coding: utf-8


from .page_base import (
    TaskHandlerBase, TextProcessMixin, 
    SCRIPT_BASE_HEAD, SCRIPT_BASE_TAIL, STYLE_BASE,
    STYLE_TEXTBOX, STYLE_CHECKBOX
)

    
class TTextExtraction(TaskHandlerBase, TextProcessMixin):
    
    def get_html_text(self, _id):
        
        fields = self.compile_fields(_id)
        text = self.render_text(**fields)
        return text
    
    def compile_fields(self, _id):
        
        data = self.data[_id]
        has_answers = len(data['answers']['text']) > 0
        metadata = self.metadata
        fields = {}
        
        if has_answers:
            fields['gold_actions'] = 'move_to+#input_answer click key_stroke+answer move_to+#submit click'
            fields['answer'] = data['answers']['text'][0]
        else:
            fields['gold_actions'] = 'move_to+#cb_unanswerable click move_to+#submit click'
            fields['answer'] = ''
        
        fields['instruction'] = metadata['instructions'][0]
        fields['paragraph'] = data['context']
        fields['question'] = data['question']
        
        fields['surfaces'] = {
            'context': metadata['context_surfaces'][0],
            'question': metadata['question_surfaces'][0],
            'unanswerable': metadata['unanswerable_surfaces'][0],
        }
        
        return fields
    
    def render_text(self, instruction, surfaces, paragraph, question, answer, gold_actions):
        
        # Font settings for this tasks
        font_size_head, font_size_main, font_size_tail, line_height = ('16px', '16px', '16px', 1.4)
        
        # Escaping
        # we do not need escape except for newline for target sentences of emulated ocr.
        instruction = self.escape_new_line(instruction)
        question = self.escape_new_line(question)
        paragraph = self.escape_new_line(paragraph)
        context_surface = self.escape_new_line(surfaces['context'])
        question_surface = self.escape_new_line(surfaces['question'])
        unanswerable_surface = self.escape_new_line(surfaces['unanswerable'])
        
        style = 'padding: 5px; border-style: solid;'
        context_p = f'<div style="{style}"><strong>{context_surface}:</strong> {paragraph}<br/></div>'
            
        style = 'padding: 5px; border-style: solid;'
        context_q = f'<div style="{style}"><strong>{question_surface}:</strong> {question}<br/></div>'
        
        # html
        src = '<!DOCTYPE html>'
        src += '<html style="overflow-y: scroll;">'
            
        # head
        src += '<head>'
        src += '<meta name="status" content=""><meta name="submitted" content="">'
        src += f'<meta name="teacher_actions" content="{gold_actions}">'
        src += f'<meta name="answer" content="{self.escape_double_quote(answer)}">'
        src += '<script>'+SCRIPT_BASE_HEAD+'</script>'
        src += '<style>'
        src += STYLE_BASE
        src += STYLE_TEXTBOX
        src += STYLE_CHECKBOX
        src += '</style>'
        src += '</head>'
            
        # body
        src += '<body>'
        
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_head}; line-height:{line_height};'
        src += f'<div class="div_head" style="{style}">{instruction}</div>'
        
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_main}; line-height:{line_height};'
        src += f'<div class="div_main" style="{style}">{context_p}</div>'
        src += f'<div class="div_main" style="{style}">{context_q}</div>'
        
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_tail}; line-height:{line_height};'
        src += f'<div class="div_tail" style="{style}">'
        src += '<div style="margin-bottom: 10px;">Your answer: <br/></div>'
        src += '<form action="javascript:void(0);" style="display:inline-block;" autocomplete="off">'
        src += '<input type="text" id="input_answer" style="width: 200px;" /> '
        
        src += '<label class="mycheckbox">'
        src += '<input type="checkbox" value="unanswerable" id="unanswerable" />'
        src += f'<span class="checkmark"  id="cb_unanswerable"></span> {unanswerable_surface} '
        src += '</label>'
            
        onclick = 'task_submit({unanswerable:document.getElementById(\'unanswerable\').checked, ' + \
                    'answer:document.getElementById(\'input_answer\').value})'
        src += f'<button id="submit" style="vertical-align: bottom; padding: 4px 8px;" onclick="{onclick}">submit</button>'
        src += '</form>'
        src += '</div>'
            
        src += '<script>'+SCRIPT_BASE_TAIL+'</script>'
        src += '</body></html>'
            
        src = self.emulated_ocr(src, self.is_debugging)
        return src
