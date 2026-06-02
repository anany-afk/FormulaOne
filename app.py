from flask import Flask, request, Response, send_from_directory, jsonify
import subprocess
import sys
import os
import uuid
import json

app = Flask(__name__, static_folder=".")

OUTPUTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "outputs")
)
os.makedirs(OUTPUTS_ROOT, exist_ok=True)


print("=" * 80)
print("F1 Telemetry Intelligence System")
print("APP FILE:", os.path.abspath(__file__))
print("OUTPUTS ROOT:", OUTPUTS_ROOT)
print("=" * 80)
sys.stdout.flush()


@app.route("/")
def index():
    return send_from_directory(".", "F1_dashboard.html")


@app.route("/run", methods=["POST"])
def run():

    print("\n" + "=" * 80)
    print("[FLASK] /run endpoint hit")
    print("=" * 80)
    sys.stdout.flush()

    data = request.get_json()

    print("[FLASK] Request JSON:")
    print(json.dumps(data, indent=2))
    sys.stdout.flush()

    circuit = data.get("circuit", "Suzuka")
    season = str(data.get("season", "2025"))
    session_type = data.get("session_type", "R")
    driver1 = data.get("driver1", "VER")
    driver2 = data.get("driver2", "NOR")

    run_id = str(uuid.uuid4())[:8]

    output_dir = os.path.abspath(
        os.path.join(OUTPUTS_ROOT, run_id)
    )

    os.makedirs(output_dir, exist_ok=True)
    print("[FLASK] CREATED:", output_dir)
    print("[FLASK] EXISTS:", os.path.exists(output_dir))
    sys.stdout.flush()

    print("[FLASK] RUN ID:", run_id)
    print("[FLASK] OUTPUT DIR:", output_dir)
    print("[FLASK] OUTPUT EXISTS:", os.path.exists(output_dir))
    sys.stdout.flush()

    script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "F1_main_headless.py")
    )

    cmd = [
        sys.executable,
        "-u",
        script,
        circuit,
        season,
        session_type,
        driver1,
        driver2,
        output_dir
    ]

    print("\n[FLASK] COMMAND:")
    print(" ".join(cmd))
    print()
    sys.stdout.flush()

    def generate():

        yield f"data: {json.dumps({'type': 'run_id', 'run_id': run_id})}\n\n"

        try:

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in proc.stdout:

                line = line.rstrip()

                print("[SUBPROCESS]", line)
                sys.stdout.flush()

                if line:
                    yield f"data: {json.dumps({'type': 'log', 'text': line})}\n\n"

            proc.wait()
            print("[FLASK] AFTER PROCESS")
            print("[FLASK] output_dir =", output_dir)
            print("[FLASK] exists =", os.path.exists(output_dir))

            if os.path.exists(output_dir):
                print("[FLASK] contents =", os.listdir(output_dir))
            else:
                print("[FLASK] DIRECTORY MISSING")

            print("\n[FLASK] PROCESS FINISHED")
            print("[FLASK] RETURN CODE:", proc.returncode)

            print("\n[FLASK] DIRECTORY WALK:")
            for root, dirs, files in os.walk(output_dir):

                print("DIR:", root)

                for f in files:

                    full = os.path.join(root, f)

                    try:
                        size = os.path.getsize(full)
                    except:
                        size = -1

                    print(f"   FILE: {f} ({size} bytes)")

            print("=" * 80)
            sys.stdout.flush()

            if proc.returncode == 0:
                yield f"data: {json.dumps({'type': 'done', 'run_id': run_id})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'text': f'Process exited with code {proc.returncode}'})}\n\n"

        except Exception as e:

            print("[FLASK ERROR]", str(e))
            sys.stdout.flush()

            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/plots/<run_id>")
def list_plots(run_id):

    run_dir = os.path.join(OUTPUTS_ROOT, run_id)

    print("[PLOTS REQUEST]", run_dir)

    if not os.path.exists(run_dir):
        print("[PLOTS] DIRECTORY DOES NOT EXIST")
        return jsonify([])

    files = [
        f for f in os.listdir(run_dir)
        if f.endswith(".png")
    ]

    print("[PLOTS] FOUND:", files)

    return jsonify(sorted(files))


@app.route("/plots/<run_id>/<filename>")
def serve_plot(run_id, filename):

    run_dir = os.path.join(OUTPUTS_ROOT, run_id)

    print("[IMAGE REQUEST]")
    print("DIR:", run_dir)
    print("FILE:", filename)

    return send_from_directory(run_dir, filename)


@app.route("/debug")
def debug():

    result = []

    for root, dirs, files in os.walk(OUTPUTS_ROOT):

        result.append({
            "root": root,
            "files": files
        })

    return jsonify(result)


if __name__ == "__main__":

    app.run(
        debug=False,
        port=5000,
        threaded=True
    )