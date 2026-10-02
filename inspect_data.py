import pyarrow as pa
import pyarrow.ipc
import json
import csv
import os

def extract_entities_from_sample(words, labels):
    entities = []
    current_entity = None

    for i, (word, label) in enumerate(zip(words, labels)):
        if label.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = {
                "text": word,
                "type": entity_type,
                "start_idx": i,
                "end_idx": i,
                "words": [word],
            }
        elif label.startswith("I-") and current_entity is not None:
            entity_type = label[2:]
            if entity_type == current_entity["type"]:
                current_entity["text"] += " " + word
                current_entity["end_idx"] = i
                current_entity["words"].append(word)
            else:
                entities.append(current_entity)
                current_entity = {
                    "text": word,
                    "type": entity_type,
                    "start_idx": i,
                    "end_idx": i,
                    "words": [word],
                }
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    return entities

def main():
    arrow_path = r"E:\re implement\cache\leduckhai___viet_med-ner\default\0.0.0\e3d0393c733858402a7c04228f45d351d2ce6d8f\viet_med-ner-test.arrow"
    
    print(f"Reading from: {arrow_path}")
    with pa.OSFile(arrow_path, 'rb') as f:
        with pa.ipc.open_stream(f) as reader:
            table = reader.read_all()
            
    print()
    print("=" * 70)
    print("  DATASET INFO")
    print("=" * 70)
    print(f"  So samples: {table.num_rows}")
    print(f"  Columns:    {table.column_names}")
    print("=" * 70)

    N = min(10, table.num_rows)
    os.makedirs("./outputs", exist_ok=True)

    json_rows = []
    for i in range(N):
        # We need to extract basic text cols
        words = table.column("words")[i].as_py()
        labels = table.column("labels")[i].as_py()
        entities = extract_entities_from_sample(words, labels)
        
        row = {"index": i}
        for col in table.column_names:
            if col == "audio":
                # audio is a struct, let's just extract path
                audio_dict = table.column("audio")[i].as_py()
                row["audio_path"] = audio_dict.get("path", "") if isinstance(audio_dict, dict) else ""
            else:
                row[col] = table.column(col)[i].as_py()
                
        row["entities_extracted"] = entities
        json_rows.append(row)

    json_path = "./outputs/vietmed_ner_samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON saved: {json_path} ({N} samples)")

    csv_path = "./outputs/vietmed_ner_samples.csv"
    text_cols = [c for c in table.column_names if c != "audio"]
    csv_fields = [
        "index", "text", "duration", "num_words",
        "words_preview", "labels_preview",
        "num_entities", "entities_text",
        "audio_path"
    ]
    extra_cols = [c for c in text_cols if c not in ("words", "labels", "text", "duration")]
    csv_fields.extend(extra_cols)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for i in range(N):
            words = table.column("words")[i].as_py()
            labels = table.column("labels")[i].as_py()
            entities = extract_entities_from_sample(words, labels)

            audio_dict = table.column("audio")[i].as_py()
            audio_path = audio_dict.get("path", "") if isinstance(audio_dict, dict) else ""

            row = {
                "index": i,
                "text": table.column("text")[i].as_py(),
                "duration": table.column("duration")[i].as_py() if "duration" in table.column_names else "",
                "num_words": len(words),
                "words_preview": " | ".join(words[:15]),
                "labels_preview": " | ".join(labels[:15]),
                "num_entities": len(entities),
                "entities_text": " ;; ".join(f"[{e['type']}] {e['text']}" for e in entities),
                "audio_path": audio_path,
            }
            for col in extra_cols:
                row[col] = table.column(col)[i].as_py()
            writer.writerow(row)

    print(f"  CSV saved:  {csv_path} ({N} samples)")

if __name__ == "__main__":
    main()
