# coding: utf-8
# Provide base classes for the task pages


import tornado.web
import tornado.escape
import re


# CSS templates
STYLE_BASE="""body {font-family: "DejaVu Serif", serif;}
button {font-family: "DejaVu Serif", serif;font-size: 13px;}
input[type=text] {font-family: "DejaVu Serif", serif; font-size: 13px;}
"""

# To make form item's appearances the same in different browsers
STYLE_TEXTBOX="""input[type=text] {padding: 4px; border-radius: 3px; border: 2px solid #ddd; box-sizing: border-box; caret-color: black;}
input[type=text]:focus {border: 2px solid #6495ED; z-index: 10; outline: 0; caret-color: black;}
"""

# Reserve mycheckbox and checkmark classes
STYLE_CHECKBOX=""".mycheckbox {display: inline-block; position: relative; margin: 0 1em; 
    padding-left: 25px; cursor: pointer; user-select: none;}
.mycheckbox input {display: none;}
.checkmark {position: absolute; top: 0; left: 0; height: 20px; width: 20px; 
    border: solid 2px #d4dae2; border-radius: 4px; box-sizing: border-box;}
.checkmark:after {content: ""; position: absolute; left: 4px; top: 1px; width: 6px; height: 10px; 
    border: solid #FFF; border-width: 0 2px 2px 0; transform: rotate(45deg); opacity: 0;}
.mycheckbox input:checked + .checkmark {background: #2e80ff; border-color: #2e80ff;}
.mycheckbox input:checked + .checkmark:after {opacity: 1;}
"""


# Scripts
# Emulated OCR related script to be embedded in the task pages
SCRIPT_BASE_HEAD="""// Define Global Functions
// View Updation Timing
var update_time = (new Date()).getTime()
function view_updated(_){
  update_time = (new Date()).getTime()
}

// Timestamp Aggregation
function get_latest_update() {
    times = Array.from(document.getElementsByTagName('iframe')).filter(
        elem => 'get_latest_update' in elem.contentWindow
    ).map(elem => elem.contentWindow.get_latest_update())
    times.push(update_time)
    return Math.max(...times)
}

// OCR Emulation
var words_cache = null
var cache_timestamp = null
function get_displayed_words(bias_x=0, bias_y=0, requires_sort=true) {

  // Filter
  const thresh = 0.4
  const iw = window.innerWidth
  const ih = window.innerHeight  

  // Children
  words = Array.from(document.getElementsByTagName('iframe')).filter(
        elem => 'get_displayed_words' in elem.contentWindow
  ).map(elem => {
      const rect = elem.getBoundingClientRect()
      return elem.contentWindow.get_displayed_words(rect.x, rect.y, false)
  }).flat().filter(w => w.x >= -thresh*w.w && w.y >= -thresh*w.h && w.x + (1-thresh)*w.w <= iw && w.y + (1-thresh)*w.h <= ih)
    
  // Add words In this frame
  if (cache_timestamp === null || cache_timestamp < update_time){
      words_cache = Array.from(document.getElementsByClassName('w')).map(elem => {
          const rect = elem.getBoundingClientRect()
          return {s:elem.innerText, x:rect.x, y:rect.y, w:rect.width, h:rect.height}
      }).filter(w => w.x >= -thresh*w.w && w.y >= -thresh*w.h && w.x + (1-thresh)*w.w <= iw && w.y + (1-thresh)*w.h <= ih)
      cache_timestamp = update_time
  }
  words = words.concat(words_cache)
  
  // Offset
  words = words.map(elem => {return {s:elem.s, x:elem.x+bias_x, y:elem.y+bias_y, w:elem.w, h:elem.h}})
  
  // Sort
  if (requires_sort) {
      words.sort((a, b) => a.y - b.y || a.x - b.x)
  }
  return words
}

// check in screen for iframe's item
function is_in_screen(iframe_id, query){
    target_iframe = document.getElementById(iframe_id)
    iframe_rect = target_iframe.getBoundingClientRect()
    rect = target_iframe.contentDocument.querySelector(query).getBoundingClientRect()
    displayed_all = (rect.x  >= 0 && rect.y >= 0 && iframe_rect.width >= rect.x + rect.width && iframe_rect.height >= rect.y + rect.height)
    return {displayed_all:displayed_all, x:rect.x+iframe_rect.x, y:rect.y+iframe_rect.y, w:rect.width, h:rect.height}
}

// check scroll position
function is_frame_scrolled_at_end(iframe_id){
    elem = document.getElementById(iframe_id).contentDocument.scrollingElement
    return elem.clientHeight + elem.scrollTop - elem.scrollHeight >= 0
}

// Task Submission
function task_submit(data){
  document.head.querySelector("meta[name='submitted']").content = JSON.stringify(data)
  document.head.querySelector("meta[name='status']").content = 'submitted'
}
"""

SCRIPT_BASE_TAIL="""window.addEventListener('resize', view_updated);
window.addEventListener('load', view_updated);
window.addEventListener('scroll', view_updated);
var dom_observer = new MutationObserver(view_updated);
dom_observer.observe(document.body, {attributes: true, childList: true, subtree: true});
"""


class TaskHandlerBase(tornado.web.RequestHandler):
    
    def initialize(self, **kwargs):
        
        self.is_debugging = False
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def get(self):
        
        _id = self.get_argument('id', None)
        if _id is None:
            text = self.get_metadata()
        else:
            text = self.get_html_text(int(_id))
        
        self.write(text)
       
    def get_metadata(self):
        """Currently, just returns the number of the data"""
        num = len(self.data)
        return str(num)
    
    def get_html_text(self, _id):
        raise NotImplementedError
        

class TextProcessMixin:
    """A mixin that provides a standatd emulated ocr."""
    
    @staticmethod
    def escape_new_line(s):
        
        return s.replace('\n', '<br/>')
    
    @staticmethod
    def escape_double_quote(s):
        
        return s.replace('"', '&quot;')
    
    @staticmethod
    def escape_all(s):
        
        return s.replace('"', '&quot;').replace('\n', '<br/>')
    
    @staticmethod
    def insert_spans(text, style=None):
        """Logic that inserts span tasgs into a given text."""
        
        if style is None:
            _tmp = '<span class="w">%s</span>'
        else:
            _tmp = '<span class="w" style="%s">%s</span>'%(style, '%s')
        
        words = filter(lambda x:x, re.split(r'(\s+|\-+)', text))
         
        surface = []
         
        re_not_a_space = re.compile(r'\S+')
        for w in words:
            if re_not_a_space.search(w):
                surface.append(_tmp%w)
            else:
                surface.append(w)
        
        return ''.join(surface)
    
    @classmethod
    def emulated_ocr(cls, html_src, is_debugging):
        """Insert spans while escaping the HTML segments that are not displayed"""
        
        re_option = re.MULTILINE + re.DOTALL + re.IGNORECASE
            
        m_body = re.search('<body>.+</body>', html_src, re_option)
        if m_body is None:
                return html_src
            
        start_pos, end_pos = m_body.span()
            
        if is_debugging:
            func = lambda t: cls.insert_spans(tornado.escape.xhtml_escape(t), 
                                              style='border-style: solid; border-width: 1px;')
        else:
            func = lambda t: cls.insert_spans(tornado.escape.xhtml_escape(t))
        
        outputs = []
        outputs.append(html_src[:start_pos])
        
        target_text = m_body.group(0)
        last_end = 0
        for m in re.finditer('</?[a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+\s*=\s*"[^"]*")*\s*/?>', target_text):
            cur_start, cur_end = m.span() 
            if last_end < cur_start:
                tag = m.group(0).strip('</>')
                content = target_text[last_end:cur_start]
                if tag not in ['script', 'style']:
                    content = re.findall('^(\s*)(.*[^\s])(\s*)$', content, re_option)
                    if content:
                        l_space, words, r_space = content[0]
                        outputs.append(l_space + func(words) + r_space)
                else:
                    outputs.append(content)
            outputs.append(target_text[cur_start:cur_end])
            last_end = cur_end
            
        outputs.append(html_src[end_pos:])
        
        return ''.join(outputs)
