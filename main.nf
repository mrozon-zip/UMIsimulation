#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*********************************************************************
 * Helper functions:
 *  - parseInstructionFile(file) reads the instructions text file,
 *    grouping lines into sections (e.g. "generate", "amplify_pcr",
 *    "amplify_polonies").
 *
 *  - buildArgs(map) creates a string of command line arguments. It
 *    skips any parameter with a value of "None". For keys
 *    "simulate" or "no_simulate", if the value is "True" (case-insensitive)
 *    it adds the flag without a value.
 *********************************************************************/
def parseInstructionFile(String filePath) {
    def blocks = [:]
    def currentBlock = null
    new File(filePath).eachLine { line ->
        line = line.trim()
        if( line ) {
            // If the line does not contain a comma, assume it is a block header.
            if( !line.contains(',') ) {
                currentBlock = line
                blocks[currentBlock] = [:]
            }
            else if( currentBlock != null ) {
                def parts = line.split(",",2)
                def key = parts[0].trim()
                def value = parts[1].trim()
                blocks[currentBlock][key] = value
            }
        }
        else {
            currentBlock = null
        }
    }
    return blocks
}

def buildArgs(Map params) {
    def args = []
    params.each { k, v ->
        if( v == "None" ) return  // skip parameters with value "None"
        if( k in ["simulate", "no_simulate"] ) {
            if( v.toLowerCase() == "true" ) {
                args << "--${k}"   // flag with no value
            }
        }
        else {
            args << "--${k}" << "${v}"
        }
    }
    return args.join(" ")
}

/*********************************************************************
 * Parameters:
 *   - instructions: path to your instructions text file.
 *   - input_csv: optionally provide an existing CSV file.
 *********************************************************************/
params.instructions = params.instructions ?: 'instructions.txt'
params.input_csv    = params.input_csv ?: null

// Parse the instructions file into a map of blocks.
def instr = parseInstructionFile(params.instructions)

// Create a global channel for the CSV input that will feed the downstream processes.
input_csv_ch = Channel.create()

/*********************************************************************
 * Process: GENERATE
 *
 * If the user has not provided a CSV file, this process runs the
 * generate step using the parameters from the "generate" block and
 * produces a CSV file.
 *
 * The output file name is taken from the instructions file “output”
 * parameter if given (otherwise it defaults to "generated.csv").
 *********************************************************************/
process GENERATE {
    tag "Generate CSV file"
    publishDir "generate_out", mode: 'copy'

    output:
      file(genOutput) into gen_out

    script:
      // Determine output file name from the instruction block (default if not given)
      def genOutput = instr.generate.containsKey("output") ? instr.generate.output : "generated.csv"
      // Build the argument string using the helper function.
      def argString = buildArgs(instr.generate)
      """
      python3 main.py generate ${argString} --output ${genOutput}
      """
}

/*********************************************************************
 * Process: AMPLIFY_PCR
 *
 * This process runs one instance of the amplify command for PCR.
 * It takes the CSV file (whether provided by the user or created by GENERATE)
 * as its input. The output file is stored in folder results1.
 *
 * The output filename is constructed from the block parameters. Any extra
 * parameters (other than those used to build a minimal name) are appended.
 *********************************************************************/
process AMPLIFY_PCR {
    tag "Amplify PCR"
    publishDir "results1", mode: 'copy'

    input:
      file csv_file from input_csv_ch

    output:
      file("*") into amp_pcr_ch

    script:
      def paramsBlock = instr.amplify_pcr
      def argString = buildArgs(paramsBlock)
      // Build a filename that starts with the mutation_rate, then appends any extra parameters.
      def mut = paramsBlock.mutation_rate ?: "NA"
      def extra = ""
      paramsBlock.each { k, v ->
          if( k != "mutation_rate" && v != "None" ) {
              extra += "_${k}_${v}"
          }
      }
      def outFile = "amplify_pcr_out_mut_${mut}${extra}.csv"
      """
      python3 main.py amplify --input ${csv_file} ${argString} --output ${outFile}
      """
}

/*********************************************************************
 * Process: AMPLIFY_POLONIES
 *
 * This process runs the second amplify command (for polonies). It uses
 * the parameters from the "amplify_polonies" block.
 *
 * The output file name is built using a pattern. For example:
 * results1/mut_{mutation_rate}_Sr_{S_radius}_dens_{denisty}_SP_{success_prob}_dev_{deviation}.csv
 * and if additional parameters (e.g. deletion_prob) are provided, they are appended.
 *********************************************************************/
process AMPLIFY_POLONIES {
    tag "Amplify Polonies"
    publishDir "results1", mode: 'copy'

    input:
      file csv_file from input_csv_ch

    output:
      file("*") into amp_poly_ch

    script:
      def paramsBlock = instr.amplify_polonies
      def argString = buildArgs(paramsBlock)
      def mut = paramsBlock.mutation_rate ?: "NA"
      def Sr  = paramsBlock.S_radius      ?: "NA"
      def dens = paramsBlock.denisty       ?: "NA"   // note: “denisty” per the provided instructions
      def sp  = paramsBlock.success_prob  ?: "NA"
      def dev = paramsBlock.deviation       ?: "NA"
      def outFile = "mut_${mut}_Sr_${Sr}_dens_${dens}_SP_${sp}_dev_${dev}"
      // Append any additional keys not in the base list.
      def expected = ["mutation_rate", "S_radius", "denisty", "success_prob", "deviation"]
      paramsBlock.each { k, v ->
         if( !(k in expected) && v != "None" ) {
             outFile += "_${k}_${v}"
         }
      }
      outFile += ".csv"
      """
      python3 main.py amplify --input ${csv_file} ${argString} --output ${outFile}
      """
}

/*********************************************************************
 * Process: DENOISE
 *
 * This process takes the output from each amplify process (PCR and polonies)
 * and runs the denoise command. Since denoise accepts only one input at a time,
 * we merge the two amplify outputs and run DENOISE once per file.
 *
 * The output file is saved in folder results2 and is named using the base name
 * of the amplify file.
 *********************************************************************/
process DENOISE {
    tag "Denoise"
    publishDir "results2", mode: 'copy'

    input:
      file amp_file from merged_amp_ch

    output:
      file("*") into denoise_ch

    script:
      // Use the input file’s base name (without extension) to create the output filename
      def base = amp_file.getBaseName()
      def outFile = "denoised_${base}.csv"
      """
      python3 main.py denoise --input ${amp_file} --output ${outFile}
      """
}

// Create channels to collect outputs from the processes.
gen_out      = Channel.create()
amp_pcr_ch   = Channel.create()
amp_poly_ch  = Channel.create()
denoise_ch   = Channel.create()
// Merge the two amplify channels so that each output gets processed by DENOISE.
merged_amp_ch = Channel.merge(amp_pcr_ch, amp_poly_ch)

/*********************************************************************
 * Workflow:
 *
 * 1. If the user has provided an input CSV file (via --input_csv), use that.
 *    Otherwise, run the GENERATE process.
 *
 * 2. The CSV file (from GENERATE or provided) is then passed into both
 *    amplification processes.
 *
 * 3. Their outputs (from amplify_pcr and amplify_polonies) are merged and
 *    sent to the DENOISE process, which runs once per amplify output.
 *********************************************************************/
workflow {

    // Define the CSV channel: if the user provided a CSV file, use it;
    // otherwise wait for the output of the GENERATE process.
    def csv_ch = params.input_csv ? Channel.fromPath(params.input_csv) : gen_out

    // Make the CSV channel available for both amplify processes.
    csv_ch.subscribe { file ->
        // Emit the file into the global input channel.
        input_csv_ch.emit(file)
    }

    // Launch the amplification processes (they automatically receive files from input_csv_ch).
    // Their outputs are sent into amp_pcr_ch and amp_poly_ch respectively.

    // After both amplifications finish, merge their outputs for denoising.
    merged_amp_ch.subscribe() // triggers consumption by the DENOISE process

    // Finally, the DENOISE process is triggered via the merged channel.
    // (All processes run concurrently as soon as inputs are available.)
}