#!/usr/bin/env python3

import chess
import sys
import time
import threading
from collections import defaultdict

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

INF = 99999999
MATE_SCORE = 90000
DRAW_SCORE = 0

# PeSTO (Midgame) values - намного сильнее стандартных
MG_PAWN = 82
MG_KNIGHT = 337
MG_BISHOP = 365
MG_ROOK = 477
MG_QUEEN = 1025
MG_KING = 0

# PeSTO Position Tables (flipped for black handled by mirroring)
PST_MG = {
    chess.PAWN: [
        0,   0,   0,   0,   0,   0,  0,   0,
        98, 134,  61,  95,  68, 126, 34, -11,
        -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
        0,   0,   0,   0,   0,   0,  0,   0,
    ],
    chess.KNIGHT: [
        -167, -89, -34, -49,  61, -97, -15, -107,
        -73, -41,  72,  36,  23,  62,   7,  -17,
        -47,  60,  37,  65,  84, 129,  73,   44,
        -9,  17,  19,  53,  37,  69,  18,   22,
        -13,   4,  16,  13,  28,  19,  21,   -8,
        -23,  -9,  12,  10,  19,  17,  25,  -16,
        -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    chess.BISHOP: [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
        -4,   5,  19,  50,  37,  37,   7,  -2,
        -6,  13,  13,  26,  34,  12,  10,   4,
        0,  15,  15,  15,  14,  27,  18,  10,
        4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    chess.ROOK: [
        32,  42,  32,  51, 63,  9,  31,  43,
        27,  32,  58,  62, 80, 67,  26,  44,
        -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,
    ],
    chess.QUEEN: [
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
        -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
        -1, -18,  -9, -19, -30, -15, -18, -23,
    ],
    chess.KING: [
        -65,  23,  16, -15, -56, -34,   2,  13,
        29,  -1, -20,  -7,  -8,  -4, -38, -29,
        -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ]
}

# TT Constants
TT_SIZE = 1_000_000  # Adjust based on memory
TT_EXACT = 1
TT_ALPHA = 2
TT_BETA = 3

class TTEntry:
    __slots__ = ('zobrist', 'depth', 'flag', 'score', 'move')
    def __init__(self, zobrist, depth, flag, score, move):
        self.zobrist = zobrist
        self.depth = depth
        self.flag = flag
        self.score = score
        self.move = move

# =============================================================================
# ENGINE CLASS
# =============================================================================

class Engine:
    def __init__(self):
        self.tt = [None] * TT_SIZE
        self.history = defaultdict(int)
        self.killers = defaultdict(lambda: [None, None]) # [move1, move2] per depth
        self.nodes = 0
        self.start_time = 0
        self.time_limit = 0
        self.stop_event = None
        self.best_move = None
        
        # Precompute flipped PST for black
        self.pst_mg = {}
        for pt, table in PST_MG.items():
            self.pst_mg[(pt, chess.WHITE)] = table
            self.pst_mg[(pt, chess.BLACK)] = [table[chess.square_mirror(sq)] for sq in range(64)]

        self.piece_vals = {
            chess.PAWN: MG_PAWN, chess.KNIGHT: MG_KNIGHT, chess.BISHOP: MG_BISHOP,
            chess.ROOK: MG_ROOK, chess.QUEEN: MG_QUEEN, chess.KING: MG_KING
        }

    def reset(self):
        self.tt = [None] * TT_SIZE
        self.history.clear()
        self.killers.clear()
        self.nodes = 0

    # ---- EVALUATION ----
    def evaluate(self, board: chess.Board):
        if board.is_checkmate():
            return -MATE_SCORE + board.halfmove_clock
        if board.is_stalemate() or board.is_insufficient_material():
            return DRAW_SCORE

        score = 0
        # Material & PST
        for sq, piece in board.piece_map().items():
            val = self.piece_vals.get(piece.piece_type, 0)
            pst_val = self.pst_mg[(piece.piece_type, piece.color)][sq]
            
            if piece.color == chess.WHITE:
                score += val + pst_val
            else:
                score -= val + pst_val

        # Mobility/Positional Tweaks (Simplified for speed)
        # Check bonus
        if board.is_check():
            score += 20 if board.turn == chess.WHITE else -20

        return score if board.turn == chess.WHITE else -score

    def get_tt_index(self, key):
        return key % TT_SIZE

    def store_tt(self, key, depth, flag, score, move):
        idx = self.get_tt_index(key)
        # Always replace for now (simple replacement scheme)
        self.tt[idx] = TTEntry(key, depth, flag, score, move)

    def probe_tt(self, key, alpha, beta, depth):
        idx = self.get_tt_index(key)
        entry = self.tt[idx]
        if entry and entry.zobrist == key:
            if entry.depth >= depth:
                if entry.flag == TT_EXACT:
                    return entry.score, entry.move
                if entry.flag == TT_ALPHA and entry.score <= alpha:
                    return entry.score, entry.move
                if entry.flag == TT_BETA and entry.score >= beta:
                    return entry.score, entry.move
            return None, entry.move
        return None, None

    # ---- MOVE ORDERING ----
    def score_move(self, board, move, tt_move, depth):
        if move == tt_move:
            return 2000000000 # TT best move first

        score = 0
        
        # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            if victim:
                score += 10 * self.piece_vals.get(victim.piece_type, 0)
            elif board.is_en_passant(move):
                score += 10 * MG_PAWN
            
            attacker = board.piece_at(move.from_square)
            if attacker:
                score -= self.piece_vals.get(attacker.piece_type, 0)
            
            score += 1000000 # Captures before quiets
        
        else:
            # Killers
            killers = self.killers[depth]
            if move == killers[0]:
                score += 900000
            elif move == killers[1]:
                score += 800000
            else:
                # History Heuristic
                score += self.history.get((board.turn, move.from_square, move.to_square), 0)

        return score

    # ---- QUIESCENCE SEARCH ----
    def quiescence(self, board, alpha, beta):
        if (self.nodes & 2047) == 0:
            if self.stop_event.is_set() or (time.time() - self.start_time > self.time_limit):
                raise SearchAbort()

        self.nodes += 1
        stand_pat = self.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Generate captures only
        moves = []
        for m in board.generate_legal_moves(chess.BB_ALL, board.occupied_co[not board.turn]):
            moves.append(m)
        
        # Basic MVV-LVA sorting for QS
        moves.sort(key=lambda m: self.score_move(board, m, None, 0), reverse=True)

        for move in moves:
            # Delta Pruning (Safety) - если взятие все равно не дотягивает до альфы
            # (skip for simplicity to avoid bugs with promotion)
            
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha

    # ---- NEGAMAX (PVS + NMP + LMR) ----
    def negamax(self, board, depth, alpha, beta, do_null=True):
        if (self.nodes & 2047) == 0:
            if self.stop_event.is_set() or (time.time() - self.start_time > self.time_limit):
                raise SearchAbort()

        self.nodes += 1
        
        # Checkmate/Stalemate detection
        if board.is_checkmate():
            return -MATE_SCORE + board.halfmove_clock
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
            return DRAW_SCORE

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        z_key = board.zobrist_hash()
        tt_val, tt_move = self.probe_tt(z_key, alpha, beta, depth)
        if tt_val is not None:
            return tt_val

        in_check = board.is_check()

        # NULL MOVE PRUNING (NMP)
        # Если ход пропускается и у нас все равно бета-отсечение, значит позиция очень сильная.
        # Не делаем в эндшпиле (zugzwang risk) и если стоим под шахом.
        if do_null and depth >= 3 and not in_check:
            # Check for major pieces (avoid zugzwang in pawn endings)
            has_pieces = any(board.pieces(pt, board.turn) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
            if has_pieces:
                board.push(chess.Move.null())
                R = 2
                null_score = -self.negamax(board, depth - 1 - R, -beta, -beta + 1, do_null=False)
                board.pop()
                
                if null_score >= beta:
                    return beta

        moves = list(board.legal_moves)
        if not moves:
            if in_check: return -MATE_SCORE + board.halfmove_clock
            return DRAW_SCORE

        # Ordering
        moves.sort(key=lambda m: self.score_move(board, m, tt_move, depth), reverse=True)

        best_score = -INF
        best_move_local = None
        start_alpha = alpha
        
        moves_searched = 0

        for move in moves:
            board.push(move)
            
            score = -INF
            
            # PVS (Principal Variation Search) logic
            if moves_searched == 0:
                # Full search for PV move
                score = -self.negamax(board, depth - 1, -beta, -alpha, True)
            else:
                # Late Move Reduction (LMR)
                # Если ход поздний, не взятие и не шах -> ищем мельче
                reduction = 0
                if depth >= 3 and moves_searched > 3 and not board.is_capture(move) and not in_check:
                    reduction = 1
                    if moves_searched > 10: reduction = 2
                
                # Search with null window
                score = -self.negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, True)
                
                # If fail high or LMR was too aggressive, re-search full window
                if score > alpha and score < beta:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, True)
                elif score > alpha and reduction > 0:
                     # Re-search without reduction
                     score = -self.negamax(board, depth - 1, -beta, -alpha, True)

            board.pop()
            moves_searched += 1

            if score > best_score:
                best_score = score
                best_move_local = move

            if score > alpha:
                alpha = score
                # Update History
                if not board.is_capture(move):
                    self.history[(board.turn, move.from_square, move.to_square)] += depth * depth
                
                if alpha >= beta:
                    # Update Killers
                    if not board.is_capture(move):
                        self.killers[depth][1] = self.killers[depth][0]
                        self.killers[depth][0] = move
                    break # Beta Cutoff

        # Store to TT
        flag = TT_EXACT
        if best_score <= start_alpha:
            flag = TT_ALPHA
        elif best_score >= beta:
            flag = TT_BETA
        
        self.store_tt(z_key, depth, flag, best_score, best_move_local)
        
        return best_score

    # ---- MAIN SEARCH LOOP ----
    def search(self, board, limits, stop_event):
        self.stop_event = stop_event
        self.nodes = 0
        self.history.clear() # clear history for new search? optional
        
        # Time Management
        self.start_time = time.time()
        
        if "movetime" in limits:
            self.time_limit = limits["movetime"] / 1000.0
        elif "wtime" in limits:
            # Classic time management: Time / 20 + Inc / 2
            wtime = limits.get("wtime", 0)
            btime = limits.get("btime", 0)
            winc = limits.get("winc", 0)
            binc = limits.get("binc", 0)
            
            my_time = wtime if board.turn == chess.WHITE else btime
            my_inc = winc if board.turn == chess.WHITE else binc
            
            # Use ~5% of remaining time + increment, heavily buffered
            alloc = (my_time / 20) + (my_inc / 2)
            # Safety: don't go below 100ms
            self.time_limit = max(0.1, alloc / 1000.0) - 0.05 
        else:
            self.time_limit = 999999

        alpha = -INF
        beta = INF
        max_depth = limits.get("depth", 64)

        best_global = None

        for depth in range(1, max_depth + 1):
            try:
                # Aspiration Windows (optional optimization, omitted for stability)
                score = self.negamax(board, depth, alpha, beta)
                
                # Get best move from TT
                z_key = board.zobrist_hash()
                _, move = self.probe_tt(z_key, -INF, INF, depth)
                
                if move:
                    best_global = move
                    # UCI Info output
                    elapsed = time.time() - self.start_time
                    nps = int(self.nodes / elapsed) if elapsed > 0 else 0
                    score_cp = score
                    # Mate score format
                    if score > MATE_SCORE - 100:
                        mate_in = (MATE_SCORE - score + 1) // 2 + 1
                        score_str = f"mate {mate_in}"
                    elif score < -MATE_SCORE + 100:
                        mate_in = (-MATE_SCORE - score) // 2
                        score_str = f"mate -{mate_in}"
                    else:
                        score_str = f"cp {score_cp}"

                    print(f"info depth {depth} score {score_str} nodes {self.nodes} nps {nps} time {int(elapsed*1000)} pv {move.uci()}")
                    sys.stdout.flush()

                if time.time() - self.start_time > self.time_limit * 0.6:
                    # If we used > 60% of time, don't start next depth
                    break

            except SearchAbort:
                break
        
        # Final best move
        if best_global:
            print(f"bestmove {best_global.uci()}")
        else:
            # Fallback
            print(f"bestmove {list(board.legal_moves)[0].uci()}")
        sys.stdout.flush()


class SearchAbort(Exception):
    pass


# =============================================================================
# UCI PROTOCOL WRAPPER
# =============================================================================

def uci():
    engine = Engine()
    board = chess.Board()
    search_thread = None
    stop_event = threading.Event()

    print("id name Okval 2.1 Test")
    print("id author me")
    print("uciok")
    sys.stdout.flush()

    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            parts = line.strip().split()
            if not parts: continue
            
            cmd = parts[0]

            if cmd == "uci":
                print("id name Okval 2.1 Test")
                print("id author me")
                print("uciok")
            
            elif cmd == "isready":
                print("readyok")
            
            elif cmd == "ucinewgame":
                engine.reset()
                board = chess.Board()
            
            elif cmd == "position":
                idx = 1
                if parts[1] == "startpos":
                    board = chess.Board()
                    idx = 2
                elif parts[1] == "fen":
                    # Handle 'fen ... moves ...'
                    fen_parts = []
                    idx = 2
                    while idx < len(parts) and parts[idx] != "moves":
                        fen_parts.append(parts[idx])
                        idx += 1
                    board = chess.Board(" ".join(fen_parts))
                
                if idx < len(parts) and parts[idx] == "moves":
                    for mv in parts[idx+1:]:
                        board.push_uci(mv)

            elif cmd == "go":
                # Parse limits
                limits = {}
                for i in range(1, len(parts)):
                    if parts[i] in ["wtime", "btime", "winc", "binc", "movetime", "depth"]:
                        limits[parts[i]] = int(parts[i+1])
                
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join()
                
                stop_event.clear()
                search_thread = threading.Thread(target=engine.search, args=(board.copy(), limits, stop_event))
                search_thread.start()

            elif cmd == "stop":
                stop_event.set()
                if search_thread: search_thread.join()

            elif cmd == "quit":
                stop_event.set()
                if search_thread: search_thread.join()
                break

        except Exception as e:
            pass

if __name__ == "__main__":
    uci()
