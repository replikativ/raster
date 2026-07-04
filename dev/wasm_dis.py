#!/usr/bin/env python3
"""Minimal wasm disassembler for raster's emitted opcode subset — enough to see
type-mismatch sites without wabt. Usage: python3 dev/wasm_dis.py file.wasm"""
import sys, struct

OPS = {
 0x00:"unreachable",0x01:"nop",0x02:"block",0x03:"loop",0x04:"if",0x05:"else",0x0b:"end",
 0x0c:"br",0x0d:"br_if",0x0f:"return",0x10:"call",
 0x20:"local.get",0x21:"local.set",0x22:"local.tee",
 0x28:"i32.load",0x2a:"f32.load",0x2b:"f64.load",0x2c:"i32.load8_s",0x2d:"i32.load8_u",
 0x2e:"i32.load16_s",0x36:"i32.store",0x38:"f32.store",0x39:"f64.store",0x3a:"i32.store8",
 0x41:"i32.const",0x42:"i64.const",0x43:"f32.const",0x44:"f64.const",
 0x45:"i32.eqz",0x46:"i32.eq",0x47:"i32.ne",0x48:"i32.lt_s",0x4a:"i32.gt_s",0x4c:"i32.le_s",0x4e:"i32.ge_s",
 0x5b:"f32.eq",0x5c:"f32.ne",0x5d:"f32.lt",0x5e:"f32.gt",0x5f:"f32.le",0x60:"f32.ge",
 0x61:"f64.eq",0x62:"f64.ne",0x63:"f64.lt",0x64:"f64.gt",0x65:"f64.le",0x66:"f64.ge",
 0x6a:"i32.add",0x6b:"i32.sub",0x6c:"i32.mul",0x6d:"i32.div_s",0x6f:"i32.rem_s",
 0x71:"i32.and",0x72:"i32.or",0x73:"i32.xor",0x74:"i32.shl",0x75:"i32.shr_s",0x76:"i32.shr_u",
 0x8b:"f32.abs",0x8c:"f32.neg",0x8e:"f32.ceil",0x8f:"f32.floor",0x91:"f32.sqrt",
 0x92:"f32.add",0x93:"f32.sub",0x94:"f32.mul",0x95:"f32.div",0x96:"f32.min",0x97:"f32.max",
 0x99:"f64.abs",0x9a:"f64.neg",0x9c:"f64.ceil",0x9d:"f64.floor",0x9f:"f64.sqrt",
 0xa0:"f64.add",0xa1:"f64.sub",0xa2:"f64.mul",0xa3:"f64.div",0xa4:"f64.min",0xa5:"f64.max",
 0xa7:"i32.wrap_i64",0xa8:"i32.trunc_f32_s",0xaa:"i32.trunc_f64_s",
 0xb2:"f32.convert_i32_s",0xb6:"f32.demote_f64",0xb7:"f64.convert_i32_s",0xbb:"f64.promote_f32",
}
IMM_LEB = {0x0c,0x0d,0x10,0x20,0x21,0x22,0x41,0x42}
IMM_MEM = {0x28,0x2a,0x2b,0x2c,0x2d,0x2e,0x36,0x38,0x39,0x3a}
IMM_BLK = {0x02,0x03,0x04}

def leb(b, i, signed=False):
    r = sh = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7f) << sh; sh += 7
        if not (x & 0x80):
            if signed and (x & 0x40): r -= 1 << sh
            return r, i

def main(path):
    b = open(path, "rb").read()
    assert b[:4] == b"\0asm"
    i = 8
    while i < len(b):
        sid = b[i]; i += 1
        size, i = leb(b, i)
        if sid == 10:  # code section
            n, j = leb(b, i)
            for f in range(n):
                fsize, j = leb(b, j)
                end = j + fsize
                nloc, j = leb(b, j)
                locs = []
                for _ in range(nloc):
                    cnt, j = leb(b, j); vt = b[j]; j += 1
                    locs.append(f"{cnt}x{ {0x7f:'i32',0x7e:'i64',0x7d:'f32',0x7c:'f64'}.get(vt, hex(vt))}")
                print(f"func[{f}] locals: {' '.join(locs)}")
                depth = 1
                while j < end:
                    off = j; op = b[j]; j += 1
                    name = OPS.get(op, f"??0x{op:02x}")
                    imm = ""
                    if op in IMM_LEB:
                        v, j = leb(b, j, signed=(op in (0x41, 0x42)))
                        imm = str(v)
                    elif op in IMM_MEM:
                        a, j = leb(b, j); o, j = leb(b, j)
                        imm = f"align={a} off={o}"
                    elif op in IMM_BLK:
                        bt = b[j]; j += 1
                        imm = {0x40:"void",0x7f:"i32",0x7d:"f32",0x7c:"f64"}.get(bt, hex(bt))
                    elif op == 0x43:
                        imm = str(struct.unpack("<f", b[j:j+4])[0]); j += 4
                    elif op == 0x44:
                        imm = str(struct.unpack("<d", b[j:j+8])[0]); j += 8
                    if op in (0x05, 0x0b): depth -= 1
                    print(f"  {off:5d} {'  '*max(depth,0)}{name} {imm}")
                    if op in (0x02, 0x03, 0x04, 0x05): depth += 1
            i += size
        else:
            i += size

main(sys.argv[1])
