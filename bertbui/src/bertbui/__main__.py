# coding: utf-8


if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) < 2:
        raise RuntimeError('Not specified command')

    cmd = sys.argv.pop(1)
    
    if cmd == 'metadata':
        from bertbui.metadata_for_static import main
        main()
    
    elif cmd == 'check_env':
        from bertbui.check_env import main
        main()
    
    elif cmd == 'record':
        from bertbui.record import main
        main()
    
    else:
        raise RuntimeError('Unknown comannd %s' % cmd)
