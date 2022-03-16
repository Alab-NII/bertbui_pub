# coding: utf-8


from .page_base import (
    TaskHandlerBase, TextProcessMixin, 
    SCRIPT_BASE_HEAD, SCRIPT_BASE_TAIL, STYLE_BASE,
    STYLE_TEXTBOX, STYLE_CHECKBOX
)


class TVisualQuestionAnswering(TaskHandlerBase, TextProcessMixin):
    
    DEFAULT_INSTRUCTION = 'Visual Question Answering Task\n' + \
            'See the picture below and answer the following question.'
    
    def get_html_text(self, _id):
        
        fields = self.compile_fields(_id)
        text = self.render_text(**fields)
        return text    
    
    def compile_fields(self, _id):
        
        data = self.data[_id]
        metadata = self.metadata
        fields = {}
        
        try:
            fields['instruction'] = metadata['instructions'][0]
        except:
            fields['instruction'] = self.DEFAULT_INSTRUCTION
        
        fields['gold_actions'] = 'move_to+#input_answer click key_stroke+answer move_to+#submit click'
        
        fields['image_path'] = data['image_path']
        fields['question'] = data['question']
        fields['answer'] = data['answer']
        
        fields['surfaces'] = {
            'question': metadata['question_surfaces'][0],
            'context': metadata['context_surfaces'][0],
        }
        
        return fields

    def render_text(self, instruction, surfaces, image_path, question, answer, gold_actions):
            
        font_size_head, font_size_main, font_size_tail, line_height = ('16px', '16px', '16px', 1.4)
            
        # Escaping
        # we do not need escape except for newline for target sentences of emulated ocr.
        instruction = self.escape_new_line(instruction)
        question = self.escape_new_line(question)
        question_surface = self.escape_new_line(surfaces['question'])
        context_surface = self.escape_new_line(surfaces['context'])
            
        style = 'display: block; margin-left: auto; margin-right: auto;'
        context_p = f'<img src="{image_path}" width="60%" style="{style}"/> <br/>'
        if context_surface:
            context_p = f'<strong>{context_surface}</strong><br/>' + context_p
            
        context_q = f'<strong>{question_surface}</strong> {question} <br/>'
            
        src = '<!DOCTYPE html>'
        src += '<html style="overflow-y: scroll;">'
        
        src += '<head>'
        src += '<meta name="status" content=""><meta name="submitted" content="">'
        src += f'<meta name="teacher_actions" content="{gold_actions}">'
        src += f'<meta name="answer" content="{self.escape_double_quote(answer)}">'
        src += '<style>'
        src += STYLE_BASE
        src += STYLE_TEXTBOX
        src += '</style>'
        src += '<script>'+SCRIPT_BASE_HEAD+'</script>'
        src += '</head>'
        
        src += '<body>'
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_head}; line-height:{line_height};'
        src += f'<div class="div_head" style="{style}">{instruction}</div>'
            
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_main}; line-height:{line_height};'
        src += f'<div class="div_main" style="{style}">{context_p}</div>'
        src += f'<div class="div_main" style="{style}">{context_q}</div>'
            
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_tail}; line-height:{line_height};'
        src += f'<div class="div_tail" style="{style}">'
        src += '<div style="margin-bottom: 10px; margin-riight: 5px; display: inline-block;">Answer: </div>'
        src += '<form action="javascript:void(0);" style="display:inline-block;" autocomplete="off">'
        src += '<input type="text" id="input_answer" style="width: 300px;" />'
            
        onclick = 'task_submit(document.getElementById(\'input_answer\').value)'
        src += f'<button id="submit" style="vertical-align: bottom; padding: 4px 8px;" onclick="{onclick}">submit</button>'
        src += '</form>'
        src += '</div>'
        
        src += '<script>'+SCRIPT_BASE_TAIL+'</script>'
        src += '</body></html>'
        
        src = self.emulated_ocr(src, self.is_debugging)
        return src
