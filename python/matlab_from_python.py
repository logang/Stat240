import time,os 

#----------------------------------------------------------------------------------------

def run_command_wrapper(cmd):
    """
    Use the subprocess module to run a shell command and collect 
    and return any error messages that result.
    """
    import subprocess
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT,shell=True) 
        return [True, result, cmd]
    except subprocess.CalledProcessError,e:                           
        return [False, str(e) + '\nProgram output was:\n' + e.output, cmd]

#----------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Parse command line options with OptionParser
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-o", "--output-file", dest="output_filename",
                      help="Specify the output filename.")
    parser.add_option("", "--debug", dest="debug", default=False, action="store_true",
                      help="Print additional information useful in debugging.")
    parser.add_option("-f", "--function", dest="function", default=False, action="store_true",
                      help="Treat input as a function rather than a script m-file.")

    (options, args) = parser.parse_args()

    # Make sure a MATLAB script is supplied (at minimum) 
    if len(args) < 1:
        print 'You must supply a MATLAB script to run'
        sys.exit(1)

    # If no output filename is given, default to 'matlab.out'.
    output_filename = options.output_filename
    if not options.output_filename:
        output_filename = os.path.join(os.path.dirname(args[0]), 'matlab.out')

    # Generate the shell command
    if options.function:
        raise NotImplementedError('This option not yet finished.')
        cmd = 'matlab -nodesktop -nosplash -r try, '
        cmd += args[0] # input expression
        cmd += ', catch, exit(1), end, exit(0)'
        if options.header:
            cmd += '| sed -e "1,/^---header---/ d"' # current header fixed to '---header---' 
        cmd += ' &'
    else:
        cmd = 'matlab -nodesktop -nosplash <'
        cmd += args[0]
        cmd += ' &'

    # Run the command and time it.
    print 'Running MATLAB file:', args[0]
    print '\t--> Shell command used:', cmd
    tic = time.time()
    result = run_command_wrapper(cmd)
    print '\t--> Finished in', time.time()-tic, 'seconds.'
    
    # parse and save results
    print 'Saving from python results not yet implemented. Save data in MATLAB script.'

    if options.debug:
        for r in result:
            print r

    1/0
# EOF

# Notes: the below works for calling a MATLAB function from the command line.
# matlab -r "try, [out1,out2]=test_function(2,2), catch, exit(1), end, exit(0)"| sed -e '1,/^---/ d'
