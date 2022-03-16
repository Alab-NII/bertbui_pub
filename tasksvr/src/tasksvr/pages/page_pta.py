# coding: utf-8


from .page_base import (
    TaskHandlerBase, TextProcessMixin, 
    SCRIPT_BASE_HEAD, SCRIPT_BASE_TAIL, STYLE_BASE,
    STYLE_TEXTBOX, STYLE_CHECKBOX
)

class TPretraining(TaskHandlerBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderers = {
            'TaskCursor': TaskInstanceCursor,
            'TaskButton': TaskInstanceButton,
            'TaskText': TaskInstanceText,
            'TaskArea': TaskInstanceArea,
        }
    
    def get_html_text(self, _id):
        
        data = self.data[_id]
        task_type = data['task_type']
        text = self.renderers[task_type](**data).render(self.is_debugging)
        return text


class TaskInstanceCursor(TextProcessMixin):
    """keys:
    instruction answer_str
    box_cx box_cy box_width box_height
    """
        
    def __init__(self,
             instruction, box_cx, box_cy, box_width, box_height,
             *args, **kwargs
       ):
        self.instruction = instruction.replace('\n', '<br/>')
        self.box = [box_cx, box_cy, box_width, box_height]
        self.answer = f'{box_cx},{box_cy}'
        self.teacher_actions = 'move_to+#correct_box'
    
    def render(self, is_debugging=False):
        
        # default font
        font_size_head, font_size_main, font_size_tail, line_height = ('16px', '16px', '16px', 1.4)
        
        # box appearance
        elem_id = 'correct_box'
        left =  int(self.box[0])
        top = int(self.box[1])
        width = int(self.box[2])
        height = int(self.box[3])
        border = '2px solid black'
        style = f'position: absolute; top: {top}%; left: {left}%; width: {width}px; height: {height}px; border: {border}'
        main_content = f'<div id="{elem_id}" style="{style}" onmouseover="task_submit(\'{elem_id}\')"><div>'
        
        # Start HTML
        src = '<!DOCTYPE html>'
        src += '<html style="overflow-y: scroll;">'
            
        src += '<head>'
        src += '<meta name="status" content=""><meta name="submitted" content="">'
        src += f'<meta name="teacher_actions" content="{self.teacher_actions}">'
        src += f'<meta name="answer" content="{self.escape_double_quote(self.answer)}">'
        src += '<script>'+SCRIPT_BASE_HEAD+'</script>'
        src += '<style>'+STYLE_BASE+'</style>'
        src += '</head>'
        
        src += '<body>'
        # header
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_head}; line-height:{line_height};'
        src += f'<div class="div_head" style="{style}">{self.instruction}</div>'
        # main
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_main}; line-height:{line_height};'
        src += f'<div class="div_main" style="{style}">{main_content}</div>'
        # tail
        # no tail
        
        src += '<script>'+SCRIPT_BASE_TAIL+'</script>'
        src += '</body></html>' 
        
        src = self.emulated_ocr(src, is_debugging)
        return src


class TaskInstanceButton(TextProcessMixin):
    """keys:
    instruction answer_str
    buttons correct_id n_rows n_columns
    """
        
    def __init__(self,
             instruction, buttons, correct_id, n_rows, n_columns,
             *args, **kwargs
       ):
        self.instruction = instruction.replace('\n', '<br/>')
        self.buttons = buttons
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.correct_id = correct_id
        
        self.answer = buttons[correct_id]
        self.teacher_actions = 'move_to+#correct_button click'
    
    def render(self, is_debugging=False):
        
        # default font
        font_size_head, font_size_main, font_size_tail, line_height = ('16px', '16px', '16px', 1.4)
        
        # make button tags
        buttons = []
        for i, button in enumerate(self.buttons):
            _id = 'id="correct_button" 'if i == self.correct_id else ''
            style = 'margin: 0.2em;  padding: 3px;'
            buttons.append(f'<button style="{style}" {_id}onclick="task_submit(\'{button}\')">{button}</button>') 
        
        # place buttons
        main_content = ''
        for i in range(0, len(buttons), self.n_columns):
            main_content += ' '.join(buttons[i:i+self.n_columns]) + '<br/>'
        
        # Start HTML
        src = '<!DOCTYPE html>'
        src += '<html style="overflow-y: scroll;">'
        
        src += '<head>'
        src += '<meta name="status" content=""><meta name="submitted" content="">'
        src += f'<meta name="teacher_actions" content="{self.teacher_actions}">'
        src += f'<meta name="answer" content="{self.escape_double_quote(self.answer)}">'
        src += '<script>'+SCRIPT_BASE_HEAD+'</script>'
        src += '<style>'+STYLE_BASE+'</style>'
        src += '</head>'
        
        src += '<body>'
        # header
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_head}; line-height:{line_height};'
        src += f'<div class="div_head" style="{style}">{self.instruction}</div>'
        # main
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_main}; line-height:{line_height};'
        src += f'<div class="div_main" style="{style}">{main_content}</div>'
        # no tail
        
        src += '<script>'+SCRIPT_BASE_TAIL+'</script>'
        src += '</body></html>'
        
        src = self.emulated_ocr(src, is_debugging)
        return src


class TaskInstanceText(TextProcessMixin):
    """keys:
    instruction answer_str
    correct_texts n_rows n_columns
    """
    
    def __init__(self,
             instruction, answer_str, 
             correct_texts, n_rows, n_columns,
             *args, **kwargs
       ):
        self.instruction = instruction.replace('\n', '<br/>')
        self.correct_texts = correct_texts
        self.n_rows = n_rows
        self.n_columns = n_columns
        
        # To make dynamically
        self.answer = None
        self.teacher_actions = None
        
    def render(self, is_debugging=False):
        
        # default font
        font_size_head, font_size_main, font_size_tail, line_height = ('16px', '16px', '16px', 1.4)
        text_box_width = '200px'
        
        # make label-text tags
        elems = []
        teacher_actions = []
        answers = []
        for i, label in enumerate(self.correct_texts):
            
            elem = ''
            style = 'margin-left: 1em; margin-right: 0.5em; display: inline-block;'
            elem += f'<label style="{style}">{label} '
            elem += f'<input type="text" id="text_{i}" style="width: {text_box_width};" value=""/>'
            elem += '</label>'
            elems.append(elem)
            
            # answer (hidden)
            answers.append(f'<meta name="answer_{i}" content="{label}">')
            
            # gold action sequence
            teacher_actions.append(f'move_to+#text_{i} click key_stroke+answer_{i}')
        
        # locate elements
        main_content = ''
        for i in range(0, len(elems), self.n_columns):
            main_content += ' '.join(elems[i:i+self.n_columns]) + '<br/>'
        
        teacher_actions.append('move_to+#submit click')
        self.teacher_actions = ' '.join(teacher_actions)
        self.answer = '+'.join(self.correct_texts)
        answers = ''.join(answers)
        
        # Start HTML
        src = '<!DOCTYPE html>'
        src += '<html style="overflow-y: scroll;">'
        
        src += '<head>'
        src += '<meta name="status" content=""><meta name="submitted" content="">'
        src += f'<meta name="teacher_actions" content="{self.teacher_actions}">'
        src += f'<meta name="answer" content="{self.escape_double_quote(self.answer)}">'
        src += answers
        src += '<style>'
        src += STYLE_BASE
        src += STYLE_TEXTBOX
        src += STYLE_CHECKBOX
        src += '</style>'
        src += '<script>'
        src += 'function submit_click(sender) {'
        src += 'const ts = []; for (var i = 0; i < %d; i++)' % len(self.correct_texts)
        src += '{ts.push(document.getElementById(\'text_\'+i).value)} task_submit(ts.join(\'+\'))'
        src += '}'
        src += SCRIPT_BASE_HEAD
        src += '</script>'
        src += '</head>'
        
        src += '<body>'
        # head
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_head}; line-height:{line_height};'
        src += f'<div class="div_head" style="{style}">{self.instruction}</div>'
        # main
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_main}; line-height:{line_height};'
        src += f'<div class="div_main" style="{style}">{main_content}</div>'
        # tail
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_tail}; line-height:{line_height};'
        src += f'<div class="div_tail" style="{style}">'
        # join contents of all the text boxes
        src += f'<button id="submit" style="padding: 4px 8px;" onclick="submit_click(this)">submit</button>'
        src += '</div>'
        
        src += '<script>'+SCRIPT_BASE_TAIL+'</script>'
        src += '</body></html>'
        
        src = self.emulated_ocr(src, is_debugging)
        return src
        

class TaskInstanceArea(TextProcessMixin):
    """keys:
    instruction answer_str
    buttons correct_id n_rows n_columns v_offset
    """
        
    def __init__(self,
             instruction, buttons, correct_id, n_rows, n_columns, v_offset,
             *args, **kwargs
       ):
        self.instruction = instruction.replace('\n', '<br/>')
        self.buttons = buttons
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.correct_id = correct_id
        self.v_offset = v_offset
        
        self.answer = buttons[correct_id]
        self.teacher_actions = 'move_to+#correct_button click'
    
    def render(self, is_debugging=False):
        
        # default font
        font_size_head, font_size_main, font_size_tail, line_height = ('16px', '16px', '16px', 1.4)
        
        # make button tags
        buttons = []
        for i, button in enumerate(self.buttons):
            _id = 'id="correct_button" 'if i == self.correct_id else ''
            style = 'margin: 0.2em;  padding: 3px;'
            buttons.append(f'<button style="{style}" {_id}onclick="task_submit(\'{button}\')">{button}</button>') 
        
        # place buttons
        main_content = ''
        for i in range(0, len(buttons), self.n_columns):
            main_content += ' '.join(buttons[i:i+self.n_columns]) + '<br/>'
        
        # Start HTML
        src = '<!DOCTYPE html>'
        src += '<html style="overflow-y: scroll;">'
        
        src += '<head>'
        src += '<meta name="status" content=""><meta name="submitted" content="">'
        src += f'<meta name="teacher_actions" content="{self.teacher_actions}">'
        src += f'<meta name="answer" content="{self.escape_double_quote(self.answer)}">'
        src += '<script>'+SCRIPT_BASE_HEAD+'</script>'
        src += '<style>'+STYLE_BASE+'</style>'
        src += '</head>'
        
        src += '<body>'
        # head
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_head}; line-height:{line_height};'
        src += f'<div class="div_head" style="{style}">{self.instruction}</div>'
        # main
        style = f'display:inline-block; float:left; clear:both; margin-bottom:10px; font-size:{font_size_main}; line-height:{line_height};'
        style += 'position:relative;'
        src += f'<div class="div_main" style="{style}">{main_content}</div>'
        # no tail
        
        # change position
        src += f'<script>document.getElementsByClassName("div_main")[0].style.top = {self.v_offset}*window.innerHeight+"px"</script>'
        
        src += '<script>'+SCRIPT_BASE_TAIL+'</script>'
        src += '</body></html>'
        
        src = self.emulated_ocr(src, is_debugging)
        return src
