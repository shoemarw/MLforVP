/* This adapted from the code provided by the Championship Value 
 * Prediction website. The original code was written by Arthur
 * Perais. https://www.microarch.org/cvp1/
 * Author: Randy Shoemaker
*/

#include <fstream>
#include <iostream>
#include <vector>
#include <bitset>

using namespace std;

#define ZERO 0
#define DEADBEEF 0xdeadbeef
#define MAXNUMLOADVALS 3

enum InstClass : uint8_t
{
  aluInstClass = 0,
  loadInstClass = 1,
  storeInstClass = 2,
  condBranchInstClass = 3,
  uncondDirectBranchInstClass = 4,
  uncondIndirectBranchInstClass = 5,
  fpInstClass = 6,
  slowAluInstClass = 7,
  undefInstClass = 8
};

enum Offset
{
  vecOffset = 32,
  ccOffset = 64
};

constexpr const char * ITYPE_NAME[] = {"aluOp", "loadOp", "stOp", "condBrOp", 
                                       "uncondDirBrOp", "uncondIndBrOp", "fpOp", 
                                       "slowAluOp", "undefined" };

// Initialize all variables associated with an instruction
uint64_t              pc;         // program counter
uint8_t               iType;      // instruction type
uint64_t              effAddr;    // effective address (for load/store)
uint8_t               memSize;    // access size (for load/store)
bool                  brTaken;    // branch taken? (for branch)
uint64_t              brTarget;   // target pc (for branch) (if taken)
uint8_t               numInRegs;  // num Input Regs
std::vector<uint8_t>  inRegVec;	  // input reg names
uint8_t               numOutRegs; // num output regs
std::vector<uint8_t>  outRegVec;  // output reg names
std::vector<uint64_t> outValVec;  // output reg values (8bytes if INT) or (16bytes if SIMD) each

void reset_instr(){
	// Set all 64 bit values to zero
	pc = effAddr = brTarget = DEADBEEF;
	// Set the instruction type to undefined
	iType = InstClass::undefInstClass;
	// Set all 8 bit values to zero
	memSize = numInRegs = numOutRegs = ZERO;
	// Clear the registers
	inRegVec.clear();
	outRegVec.clear();
	outValVec.clear();

}

void read_instr(std::ifstream * d_input, uint64_t num_instrs) {

	d_input->read((char*) &pc, sizeof(pc)); // the pc
	d_input->read((char*) &iType, sizeof(iType)); // instruction type

	if (iType >= InstClass::undefInstClass) {
		cout << "----------INVALID INSTRUCTION TYPE----------";
	}

	// See if we are dealing with a load/store
	if (iType == InstClass::loadInstClass || iType == InstClass::storeInstClass) {
		d_input->read((char*) &effAddr, sizeof(effAddr)); // effective address
		d_input->read((char*) &memSize, sizeof(memSize)); // size of memory to store/load
	}

	// See if we are dealing with a branch
	if (iType == InstClass::condBranchInstClass 
		|| iType == InstClass::uncondDirectBranchInstClass 
		|| iType == InstClass::uncondIndirectBranchInstClass) {

		d_input->read((char*) &brTaken, sizeof(brTaken)); // tells if branch was taken

		// See if the branch was taken
		if (brTaken) {
			d_input->read((char*) &brTarget, sizeof(brTarget)); // target pc
		}
	}

	d_input->read((char*) &numInRegs, sizeof(numInRegs)); // num input registers

	// Read in all input register names
	for(auto i = 0; i < numInRegs; i++) {
		uint8_t inReg;
		d_input->read((char*) &inReg, sizeof(inReg)); // input reg names
		inRegVec.push_back(inReg);
	}

	d_input->read((char*) &numOutRegs, sizeof(numOutRegs)); // num output registers

	// Read in all output register names
	for(auto i = 0; i < numOutRegs; i++) {
		uint8_t outReg;
  		d_input->read((char*) &outReg, sizeof(outReg)); // output reg names
  		outRegVec.push_back(outReg);
	}

	// Read in the output register values
	for(auto i = 0; i < numOutRegs; i++) {
  		uint64_t val;
  		d_input->read((char*) &val, sizeof(val)); // output register value
  		outValVec.push_back(val);
  		// See if its a SIMD instruction that produced this output TODO
        if(outRegVec[i] >= Offset::vecOffset && outRegVec[i] != Offset::ccOffset) {
	        d_input->read((char*) &val, sizeof(val)); // Read in 8 more bytes
    		outValVec.push_back(val);
    		// remaining pieces??
    	}
    }
}


int main(int argc, char** argv) {

	cout << "FYI: \n\tUsage: ./executable <trace_name>\n";

	// Get the file name and set up a stream to the file
	const char * trace_name = argv[1];
	ifstream * d_input = new ifstream();
	d_input->open(trace_name, ios_base::in | ios_base::binary);

	// Create a csv file for the loads. Name it "<trace_name>-loads.csv"
	ofstream out_file;
	string name(argv[1]);
	name = name + "-loads-split.csv";
	out_file.open(name.c_str());

	// Write the header row of the csv file. Loads can have 1, 2, or 3 values
	out_file << "pc_fst32,pc_snd32,eff_addr_fst32,eff_addr_snd32,num_values,val0_fst32,val0_snd32,val1_fst32,val1_snd32,val2_fst32,val2_snd32\n";

	uint64_t num_instrs = 0; // Number of instructions processed
	cout << "Reading from " << trace_name << "...\n";

	// Iterate through the trace until we reach the eof. We assume the trace is valid.
	while(!d_input->eof()) {
		
		// Read an instruction.
		read_instr(d_input, num_instrs);

		// Write each load to the output file.
		if (iType == InstClass::loadInstClass) {
			// Everything is written in base 10 since its going into a neural network
			// anyway. 0xDEADBEEF is 3,735,928,559 in base 10.
			//out_file << pc << "," << effAddr << "," << unsigned(numOutRegs);
			// break the pc into two 32 bit unsigned ints
			uint32_t fst32 = (uint32_t)(pc>>32);
			uint32_t snd32 = (uint32_t)(pc);
			// write the spit pc
			out_file << fst32 << "," << snd32;
			// split the effective address
			fst32 = (uint32_t)(effAddr>>32);
			snd32 = (uint32_t)(effAddr);
			// write the spit effective address
			out_file << "," << fst32 << "," << snd32 << "," << unsigned(numOutRegs);
			// Print the values associated with the load
			for (auto i = 0; i < numOutRegs; i++) {
				fst32 = (uint32_t)(outValVec[i]>>32);
				snd32 = (uint32_t)(outValVec[i]);
				out_file << "," << fst32 << "," << snd32;
			} 
			// Print the rest, if any, as NAN
			for (auto i = numOutRegs; i < MAXNUMLOADVALS; i++) {
				out_file << ",NAN";
			}
			out_file << "\n"; // End the row
		}

      	// Reset the instruction.
      	reset_instr();

		num_instrs++;

    	if(num_instrs % 100000 == 0) {
      		cout << "\nNumber of Instructions: " << num_instrs << std::endl;
    	}
	}

	cout << "\nDone.\n"; 
	// Clean up and exit
	delete d_input;
	return 0;
}