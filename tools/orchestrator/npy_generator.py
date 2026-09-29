import sys
import os
import argparse
import csv
from vpadanalyzer.synthesis import Synthesis


CSV_HEADER = "file,area,power,delay,pda,mean_ae,mean_ae_cnn,max_ae,accuracy\n"
CSV_FILE = "results.csv"

try:
    import tools.synthesis_npy_generation.sub_xpat_circuits_generator as sub_xpat_circuits_generator
    import tools.synthesis_npy_generation.sub_x_pat_simulator as sub_x_pat_simulator
except ImportError:
    sys.path.append(os.getcwd())
    import tools.synthesis_npy_generation.sub_xpat_circuits_generator as sub_xpat_circuits_generator
    import tools.synthesis_npy_generation.sub_x_pat_simulator as sub_x_pat_simulator

def find_matching_stats(current_stats):
    if not os.path.exists(CSV_FILE):
        return None
    try:
        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                match = True
                for key in current_stats:
                    if str(row.get(key)) != str(current_stats[key]):
                        match = False
                        break
                if match:
                    acc_val = row.get("accuracy", "").strip()
                    if acc_val != "":
                        try:
                            if float(acc_val) != 0:
                                return acc_val
                        except ValueError: continue
    except Exception:
        return None
    return None

def circuits_analizer(input_path):
    area = Synthesis.area(input_path)
    power = Synthesis.power(input_path)
    delay = Synthesis.delay(input_path)
    print("Synth completed")
    return {"file": os.path.basename(input_path), "area": area, "power": power, "delay": delay}

def generate_npy_for_single_file(input_verilog, bitwidth, output_npy_path, experiment_name, heat_maps_path):
    base_filename = os.path.basename(input_verilog)
    name, ext = os.path.splitext(base_filename)
    
    # Identificativo unico usato come chiave nel CSV
    filename_key = f"{name}_{experiment_name}"

    sub_xpat_circuits_generator.generate_approx_mult_function(input_verilog, bitwidth)

    try:
        mean_ae, mean_ae_cnn, max_error = sub_x_pat_simulator.execute_save(bitwidth, output_npy_path, heat_maps_path)
        data = circuits_analizer(input_verilog)
        pda = data['area'] * data['power'] * data['delay']

        current_metrics = {
            "area": data['area'], "power": data['power'], "delay": data['delay'],
            "pda": pda, "mean_ae": mean_ae, "mean_ae_cnn": mean_ae_cnn, "max_ae": max_error
        }

        found_accuracy = find_matching_stats(current_metrics)
        
        file_exists = os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a") as f:
            if not file_exists:
                f.write(CSV_HEADER)
            acc_str = found_accuracy if found_accuracy else ""
            csv_line = f"{filename_key},{data['area']},{data['power']},{data['delay']},{pda},{mean_ae},{mean_ae_cnn},{max_error},{acc_str}\n"
            f.write(csv_line)
        
        if found_accuracy:
            print(f"[NPY GEN] Match found! Accuracy {found_accuracy} reused.")

    except Exception as e:
        print(f"[NPY GEN] Error: {e}")
        sys.exit(1)
    finally:
        target_dir = os.path.dirname(os.path.abspath(sub_xpat_circuits_generator.__file__))
        target_file = os.path.join(target_dir, "sub_x_pat_multiplier.py")

        if os.path.exists(target_file):
            os.remove(target_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_verilog")
    parser.add_argument("bitwidth", type=int)
    parser.add_argument("output_npy")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--heat-maps-path", default="heat_maps/npy_matrix/8bit_resnet_20", required=False)
    args = parser.parse_args()
    generate_npy_for_single_file(args.input_verilog, args.bitwidth, args.output_npy, args.experiment_name, args.heat_maps_path)