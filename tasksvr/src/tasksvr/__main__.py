# coding: utf-8


if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) < 2:
        raise RuntimeError('Not specified command')

    cmd = sys.argv.pop(1)
    
    if cmd == 'download':
        from tasksvr.download import main
        main()
        
    elif cmd == 'run':
        from tasksvr.server import main
        main()
    
    elif cmd == 'evaluate':
        from tasksvr.evaluate import main
        main()

    else:
        raise RuntimeError('Unknown comannd %s' % cmd)
