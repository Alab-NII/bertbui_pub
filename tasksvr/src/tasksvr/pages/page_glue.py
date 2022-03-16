# coding: utf-8


from .page_base import (
    TaskHandlerBase, TextProcessMixin, 
    SCRIPT_BASE_HEAD, SCRIPT_BASE_TAIL, STYLE_BASE
)


class TTextClassification(TaskHandlerBase, TextProcessMixin):
    
    DEFAULT_INSTRUCTION = 'Natural Language Inference Task\n' + \
                                     'Read the following sentences and ' + \
                                     'click the button of the most appliable class.'
    DEFAULT_GOLD_ACTIONS = 'move_to+#correct_button click'
    
    def get_html_text(self, _id):
        
        fields = self.compile_fields(_id)
        text = self.render_text(**fields)
        return text
    
    def compile_fields(self, _id):
        
        data = self.data[_id]
        metadata = self.metadata
        fields = {}
        
        fields['gold_actions'] = self.DEFAULT_GOLD_ACTIONS
        
        try:
            fields['instruction'] = metadata['instructions'][0]
        except:
            fields['instruction'] = self.DEFAULT_INSTRUCTION
        
        feature_name_rule = metadata['feature_name_rules'][0]
        fields['sentences'] = sentences = []
        # tuple (name, sentence)
        for f in metadata['sentence_features']:
            name = feature_name_rule.get(f, f)
            sentence = data[f]
            sentences.append((name, sentence))
        
        label_rule = metadata['label_rules'][0]
        # answer is id (int) for classification
        fields['answer'] = int(data['label'])
        # id and name
        fields['candidates'] = [(_, label_rule.get(_, str(_))) for _ in metadata['labels']]
        
        return fields
    
    def render_text(self, instruction, sentences, candidates, answer, gold_actions):
        
        # Font settings for this tasks
        font_size_head, font_size_main, font_size_tail, line_height = ('16px', '16px', '16px', 1.4)
        
        # Escaping
        # we do not need escape except for newline for target sentences of emulated ocr.
        candidates_name = [(_id, self.escape_new_line(n)) for _id, n in candidates]
        instruction = self.escape_new_line(instruction)
        sentences = [(self.escape_new_line(n), self.escape_new_line(s)) for n, s in sentences]
        
        # Sentences
        context = '<div style="padding: 5px; border-style: solid;">'
        for name, sentence in sentences:
            if name:
                sentence = f'<strong>{name}:</strong> {sentence}'
            context += sentence + '<br/>'
        context += '</div>'
        
        # Buttons for candidates
        buttons = []
        for candidate_id, condidate_name in candidates:
            if candidate_id == answer:
                _id = 'id="correct_button" '
            else:
                _id = ''
            buttons.append(f'<button style="margin-right: 20px;" {_id}onclick="task_submit(\'{candidate_id}\')">{condidate_name}</button>')
                
        footer = '<div style="margin-bottom: 10px;">Classes:<br/></div>%s'%(''.join(buttons))
            
        # Start HTML
        src = '<!DOCTYPE html>'
        src += '<html style="overflow-y: scroll;">'
            
        src += '<head>'
        src += '<meta name="status" content=""><meta name="submitted" content="">'
        src += f'<meta name="teacher_actions" content="{gold_actions}">'
        src += f'<meta name="answer" content="{str(answer)}">'
        src += '<script>'+SCRIPT_BASE_HEAD+'</script>'
        src += '<style>'+STYLE_BASE+'</style>'
        src += '</head>'
            
        src += '<body>'
        
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_head}; line-height:{line_height};'
        src += f'<div class="div_head" style="{style}">{instruction}</div>'
            
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_main}; line-height:{line_height};'
        src += f'<div class="div_main" style="{style}">{context}</div>'
            
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_tail}; line-height:{line_height};'
        src += f'<div class="div_tail" style="{style}">{footer}</div>'
        
        src += '<script>'+SCRIPT_BASE_TAIL+'</script>'
        src += '</body></html>'
            
        src = self.emulated_ocr(src, self.is_debugging)
        return src
