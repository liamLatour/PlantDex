# reconstruct_fbs.py

def main():
    # Read decoded strings (optional, for hints)
    try:
        with open("gt_strings.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Loaded {len(lines)-1} strings from gt_strings.txt")
    except FileNotFoundError:
        print("gt_strings.txt not found. Proceeding with Java class structure only.")

    # Draft schema based on Java classes
    fbs_content = '''namespace org.plantnet.offline.pojo;

table DataConfig {
  inputSize:[ushort];
  interpolation:string;
  mean:[float];
  std:[float];
  cropPct:float;
  cropMode:string;
}

table Organ {
  id:string;
  isReject:bool;
  weight:float;
  flag:ulong;
}

table Project {
  name:string;
  prod:bool;
  priority:ubyte;
}

table Species {
  id:string;
  binomial:string;
  author:string;
  genus:ushort;
  family:ushort;
  projects:[ubyte];
  gbifId:ulong;
  isIllustrated:bool;
}

table GT {
  nbResults:ubyte;
  wrongRefThreshold:float;
  minScore:float;
  minRejectScore:float;
  maxSumScore:float;
  maxRejectRank:int;
  projects:[Project];
  families:[string];
  genus:[string];
  organs:[Organ];
  species:[Species];
  noGenus:int;
  noFamily:int;
  noProject:int;
  nbRejectClasses:ulong;
  dataConfig:DataConfig;
}

root_type GT;
'''

    with open("gt.fbs", "w", encoding="utf-8") as f:
        f.write(fbs_content)
    print("Draft gt.fbs schema written to gt.fbs")

if __name__ == "__main__":
    main()