#!/usr/bin/env /usr/local/opt/python@3.9/bin/python3 -W ignore::DeprecationWarning

__author__ = "Kishen"
__email__ = "pkishensuraj@gmail.com"

import os
import argparse
import json
import logging
import math
import chess
import chess.pgn
import chess.uci
import chess.variant
import lichess.api
from lichess.format import SINGLE_PGN

# Constants
ERROR_THRESHOLD = {
    'BLUNDER': -300,
    'MISTAKE': -150,
    'DUBIOUS': -75,
}
NEEDS_ANNOTATION_THRESHOLD = 7.5
MAX_SCORE = 10000
MAX_CPL = 2000
SHORT_PV_LEN = 10
USER = "mistborn17"
STOCKFISH14 = "/opt/homebrew/bin/stockfish"
DEPTH_DEFAULT = 15
NUMBER_OF_LICHESS_GAMES = 1

# Initialize Logging Module
logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    logger.addHandler(ch)


# Uncomment this line to get EXTREMELY verbose UCI communication logging:
# logging.basicConfig(level=logging.DEBUG)


def parse_args():
    """
    Define an argument parser and return the parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog='annotator',
        description='takes chess games in a PGN file and prints '
                    'annotations to standard output')
    parser.add_argument("--file", "-f",
                        help="input PGN file",
                        metavar="file.pgn")
    parser.add_argument("--pgnoutfile", "-o",
                        help="output PGN file",
                        metavar="pgn_out_file.pgn")
    parser.add_argument("--engine", "-e",
                        help="analysis engine (default: %(default)s)",
                        default=STOCKFISH14)
    parser.add_argument("--threads", "-t",
                        help="threads for use by the engine \
                            (default: %(default)s)",
                        type=int,
                        default=1)
    parser.add_argument("--numgames", "-n",
                        help="number of lichess games, useful only in lichess mode",
                        type=int,
                        default=NUMBER_OF_LICHESS_GAMES)
    parser.add_argument("--depth", "-d",
                        help="depth for use by the engine \
                                (default: %(default)s)",
                        type=int,
                        default=DEPTH_DEFAULT)
    parser.add_argument("--output_append", "-oa",
                        action="store_true",
                        )
    parser.add_argument("--verbose", "-v", help="increase verbosity",
                        action="count")

    return parser.parse_args()


def setup_logging(args):
    """
    Sets logging module verbosity according to runtime arguments
    """
    if args.verbose:
        if args.verbose >= 3:
            # EVERYTHING TO LOG FILE
            logger.setLevel(logging.DEBUG)
            hldr = logging.FileHandler('annotator.log')
            logger.addHandler(hldr)
        elif args.verbose == 2:
            # DEBUG TO STDERR
            logger.setLevel(logging.DEBUG)
        elif args.verbose == 1:
            # INFO TO STDERR
            logger.setLevel(logging.INFO)


def eval_numeric(info_handler):
    """
    Returns a numeric evaluation of the position, even if depth-to-mate was
    found. This facilitates comparing numerical evaluations with depth-to-mate
    evaluations
    """
    dtm = info_handler.info["score"][1].mate
    cp = info_handler.info["score"][1].cp

    if dtm is not None:
        # We have depth-to-mate (dtm), so translate it into a numerical
        # evaluation. This number needs to be just big enough to guarantee that
        # it is always greater than a non-dtm evaluation.

        if dtm >= 1:
            return MAX_SCORE - dtm
        else:
            return -(MAX_SCORE + dtm)

    elif cp is not None:
        # We don't have depth-to-mate, so return the numerical evaluation (in
        # centipawns)
        return cp

    # If we haven't returned yet, then the info_handler had garbage in it
    raise RuntimeError("Evaluation found in the info_handler was "
                       "unintelligible")


def eval_human(white_to_move, info_handler):
    """
    Returns a human-readable evaluation of the position:
        If depth-to-mate was found, return plain-text mate announcement
        (e.g. "Mate in 4")
        If depth-to-mate was not found, return an absolute numeric evaluation
    """
    dtm = info_handler.info["score"][1].mate
    cp = info_handler.info["score"][1].cp

    if dtm is not None:
        return "Mate in {}".format(abs(dtm))
    elif cp is not None:
        # We don't have depth-to-mate, so return the numerical evaluation (in
        # pawns)
        return '{:.2f}'.format(eval_absolute(cp / 100, white_to_move))

    # If we haven't returned yet, then the info_handler had garbage in it
    raise RuntimeError("Evaluation found in the info_handler was "
                       "unintelligible")


def eval_absolute(number, white_to_move):
    """
    Accepts a relative evaluation (from the point of view of the player to
    move) and returns an absolute evaluation (from the point of view of white)
    """

    return number if white_to_move else -number


def winning_chances(centipawns):
    """
    Takes an evaluation in centipawns and returns an integer value estimating
    the chance the player to move will win the game

    winning chances = 50 + 50 * (2 / (1 + e^(-0.004 * centipawns)) - 1)
    """
    return 50 + 50 * (2 / (1 + math.exp(-0.004 * centipawns)) - 1)


def needs_annotation(judgment):
    """
    Returns a boolean indicating whether a node with the given evaluations
    should have an annotation added
    """

    if judgment is None:
        return False

    best = winning_chances(int(judgment["besteval"]))
    played = winning_chances(int(judgment["playedeval"]))
    delta = best - played

    return delta > NEEDS_ANNOTATION_THRESHOLD


def judge_move(board, played_move, engine, info_handler, arg_depth):
    """
    Evaluate the strength of a given move by comparing it to engine's best
    move and evaluation at a given depth, in a given board context

    Returns a judgment

    A judgment is a dictionary containing the following elements:
          "bestmove":      The best move in the position, according to the
                           engine
          "besteval":      A numeric evaluation of the position after the best
                           move is played
          "bestcomment":   A plain-text comment appropriate for annotating the
                           best move
          "pv":            The engine's primary variation including the best
                           move
          "playedeval":    A numeric evaluation of the played move
          "playedcomment": A plain-text comment appropriate for annotating the
                           played move
          "depth":         Search depth in plies
          "nodes":         Number nodes searched
    """

    judgment = {}

    # First, get the engine bestmove and evaluation
    engine.position(board)
    engine.go(depth=arg_depth)

    judgment["bestmove"] = info_handler.info["pv"][1][0]
    judgment["besteval"] = eval_numeric(info_handler)
    judgment["pv"] = info_handler.info["pv"][1]
    judgment["depth"] = info_handler.info["depth"]
    judgment["nodes"] = info_handler.info["nodes"]

    # Annotate the best move
    judgment["bestcomment"] = eval_human(board.turn, info_handler)

    # If the played move matches the engine bestmove, we're done
    if played_move == judgment["bestmove"]:
        judgment["playedeval"] = judgment["besteval"]
        judgment["playedcomment"] = judgment["bestcomment"]

    else:
        # get the engine evaluation of the played move
        board.push(played_move)
        engine.position(board)
        engine.go(depth=arg_depth)

        # Store the numeric evaluation.
        # We invert the sign since we're now evaluating from the opponent's
        # perspective
        judgment["playedeval"] = -eval_numeric(info_handler)
        # Take the played move off the stack (reset the board)
        board.pop()
        judgment["playedcomment"] = eval_human(not board.turn, info_handler)

    return judgment


def get_nags(judgment):
    """
    Returns a Numeric Annotation Glyph (NAG) according to how much worse the
    played move was vs the best move
    """

    delta = judgment["playedeval"] - judgment["besteval"]

    if delta < ERROR_THRESHOLD["BLUNDER"]:
        return [chess.pgn.NAG_BLUNDER]
    elif delta < ERROR_THRESHOLD["MISTAKE"]:
        return [chess.pgn.NAG_MISTAKE]
    elif delta < ERROR_THRESHOLD["DUBIOUS"]:
        return [chess.pgn.NAG_DUBIOUS_MOVE]
    else:
        return []


def var_end_comment(board, judgment):
    """
    Return a human-readable annotation explaining the board state (if the game
    is over) or a numerical evaluation (if it is not)
    """
    score = judgment["bestcomment"]
    depth = judgment["depth"]

    if board.is_stalemate():
        return "Stalemate"
    elif board.is_insufficient_material():
        return "Insufficient material to mate"
    elif board.can_claim_fifty_moves():
        return "Fifty move rule"
    elif board.can_claim_threefold_repetition():
        return "Three-fold repetition"
    elif board.is_checkmate():
        # checkmate speaks for itself
        return ""
    return "{}/{}".format(str(score), str(depth))


def truncate_pv(board, pv):
    """
    If the pv ends the game, return the full pv
    Otherwise, return the pv truncated to 10 half-moves
    """

    for move in pv:
        if not board.is_legal(move):
            raise AssertionError
        board.push(move)

    if board.is_game_over(claim_draw=True):
        return pv
    else:
        return pv[:SHORT_PV_LEN]


def add_annotation(node, judgment):
    """
    Add evaluations and the engine's primary variation as annotations to a node
    """
    prev_node = node.parent

    # Add the engine evaluation
    if judgment["bestmove"] != node.move:
        node.comment = judgment["playedcomment"]

    # Get the engine primary variation
    variation = truncate_pv(prev_node.board(), judgment["pv"])

    # Add the engine's primary variation as an annotation
    prev_node.add_line(moves=variation)

    # Add a comment to the end of the variation explaining the game state
    var_end_node = prev_node.variation(judgment["pv"][0]).end()
    var_end_node.comment = var_end_comment(var_end_node.board(), judgment)

    # Add a Numeric Annotation Glyph (NAG) according to how weak the played
    node.nags = get_nags(judgment)


def add_score(node, judgment):
    """
    Add evaluations and the engine's primary variation as annotations to a node
    """
    # prev_node = node.parent

    # Add the engine evaluation
    # if judgment["bestmove"] != node.move:
    node.comment = judgment["playedcomment"]

    # Add a comment to the end of the variation explaining the game state

    # var_end_node = prev_node.variation(judgment["pv"][0]).end()
    # var_end_node.comment = judgment["playedcomment"]# var_end_comment(var_end_node.board(), judgment)


def classify_fen(fen, ecodb):
    """
    Searches a JSON file with Encyclopedia of Chess Openings (ECO) data to
    check if the given FEN matches an existing opening record

    Returns a classification

    A classfication is a dictionary containing the following elements:
        "code":         The ECO code of the matched opening
        "desc":         The long description of the matched opening
        "path":         The main variation of the opening
    """
    classification = {}
    classification["code"] = ""
    classification["desc"] = ""
    classification["path"] = ""

    for opening in ecodb:
        if opening['f'] == fen:
            classification["code"] = opening['c']
            classification["desc"] = opening['n']
            classification["path"] = opening['m']

    return classification


def eco_fen(board):
    """
    Takes a board position and returns a FEN string formatted for matching with
    eco.json
    """
    board_fen = board.board_fen()
    castling_fen = board.castling_xfen()

    to_move = 'w' if board.turn else 'b'

    return "{} {} {}".format(board_fen, to_move, castling_fen)


def debug_print(node, judgment):
    """
    Prints some debugging info about a position that was just analyzed
    """

    logger.debug(node.board())
    logger.debug(node.board().fen())
    logger.debug("Played move: %s", format(node.parent.board().san(node.move)))
    logger.debug("Best move: %s",
                 format(node.parent.board().san(judgment["bestmove"])))
    logger.debug("Best eval: %s", format(judgment["besteval"]))
    logger.debug("Best comment: %s", format(judgment["bestcomment"]))
    logger.debug("PV: %s",
                 format(node.parent.board().variation_san(judgment["pv"])))
    logger.debug("Played eval: %s", format(judgment["playedeval"]))
    logger.debug("Played comment: %s", format(judgment["playedcomment"]))
    logger.debug("Delta: %s",
                 format(judgment["besteval"] - judgment["playedeval"]))
    logger.debug("Depth: %s", format(judgment["depth"]))
    logger.debug("Nodes: %s", format(judgment["nodes"]))
    logger.debug("Needs annotation: %s", format(needs_annotation(judgment)))
    logger.debug("")


def cpl(string):
    """
    Centipawn Loss
    Takes a string and returns an integer representing centipawn loss of the
    move We put a ceiling on this value so that big blunders don't skew the
    acpl too much
    """

    cpl = int(string)

    return min(cpl, MAX_CPL)


def acpl(cpl_list):
    """
    Average Centipawn Loss
    Takes a list of integers and returns an average of the list contents
    """
    try:
        return sum(cpl_list) / len(cpl_list)
    except ZeroDivisionError:
        return 0


def clean_game(game):
    """
    Takes a game and strips all comments and variations, returning the
    "cleaned" game
    """
    node = game.end()

    while True:
        prev_node = node.parent

        node.comment = None
        node.nags = []
        for variation in reversed(node.variations):
            if not variation.is_main_variation():
                node.remove_variation(variation)

        if node == game.root():
            break

        node = prev_node

    return node.root()


def game_length(game):
    """
    Takes a game and returns an integer corresponding to the number of
    half-moves in the game
    """
    ply_count = 0
    node = game.end()

    while not node == game.root():
        node = node.parent
        ply_count += 1

    return ply_count


def classify_opening(game):
    """
    Takes a game and adds an ECO code classification for the opening
    Returns the classified game and root_node, which is the node where the
    classification was made
    """
    ecopath = os.path.join(os.path.dirname(__file__), 'eco/eco.json')
    with open(ecopath, 'r') as ecofile:
        ecodata = json.load(ecofile)

        ply_count = 0

        root_node = game.root()
        node = game.end()

        # Opening classification for variant games is not implemented (yet?)
        is_960 = root_node.board().chess960
        if is_960:
            variant = "chess960"
        else:
            variant = type(node.board()).uci_variant

        if variant != "chess":
            logger.info("Skipping opening classification in variant "
                        "game: {}".format(variant))
            return node.root(), root_node, game_length(game)

        logger.info("Classifying the opening for non-variant {} "
                    "game...".format(variant))

        while not node == game.root():
            prev_node = node.parent

            fen = eco_fen(node.board())
            classification = classify_fen(fen, ecodata)

            if classification["code"] != "":
                # Add some comments classifying the opening
                node.root().headers["ECO"] = classification["code"]
                node.root().headers["Opening"] = classification["desc"]
                node.comment = "{} {}".format(classification["code"],
                                              classification["desc"])
                # Remember this position so we don't analyze the moves
                # preceding it later
                root_node = node
                # Break (don't classify previous positions)
                break

            ply_count += 1
            node = prev_node

        return node.root(), root_node, ply_count


def add_acpl(game, root_node, annotation_side):
    """
    Takes a game and a root node, and adds PGN headers with the computed ACPL
    (average centipawn loss) for each player. Returns a game with the added
    headers.
    """
    white_cpl = []
    black_cpl = []

    node = game.end()
    while not node == root_node:
        prev_node = node.parent

        side_to_play = "white" if prev_node.board().turn else "black"
        if annotation_side != "both" and annotation_side != side_to_play:
            node = prev_node
            continue

        judgment = node.comment
        delta = judgment["besteval"] - judgment["playedeval"]

        if node.board().turn:
            black_cpl.append(cpl(delta))
        else:
            white_cpl.append(cpl(delta))

        node = prev_node

    if annotation_side != "black":  # white or both
        node.root().headers["WhiteACPL"] = str(round(acpl(white_cpl)))
    if annotation_side != "white":  # black or both
        node.root().headers["BlackACPL"] = str(round(acpl(black_cpl)))

    return node.root()


def get_total_budget(arg_gametime):
    return float(arg_gametime) * 60


def get_pass1_budget(total_budget):
    return total_budget / 7


def get_pass2_budget(total_budget, pass1_budget):
    return total_budget - pass1_budget


def get_time_per_move(pass_budget, ply_count):
    return float(pass_budget) / float(ply_count)


def analyze_game(game, arg_depth, enginepath, threads):
    """
    Take a PGN game and return a GameNode with engine analysis added
    - Attempt to classify the opening with ECO and identify the root node
        * The root node is the position immediately after the ECO
        classification
        * This allows us to skip analysis of moves that have an ECO
        classification
    - Analyze the game, adding annotations where appropriate
    - Return the root node with annotations
    """

    # First, check the game for PGN parsing errors
    # This is done so that we don't waste CPU time on nonsense games
    checkgame(game)

    ###########################################################################
    # Initialize the engine
    ###########################################################################
    try:
        engine = chess.uci.popen_engine(enginepath)
    except FileNotFoundError:
        errormsg = "Engine '{}' was not found. Aborting...".format(enginepath)
        logger.critical(errormsg)
        raise
    except PermissionError:
        errormsg = "Engine '{}' could not be executed. Aborting...".format(
            enginepath)
        logger.critical(errormsg)
        raise

    engine.uci()
    info_handler = chess.uci.InfoHandler()
    engine.info_handlers.append(info_handler)
    if game.board().uci_variant != "chess" or game.root().board().chess960:
        # This is a variant game, so confirm that the engine we're using
        # supports the variant.
        if game.root().board().chess960:
            try:
                engine.options["UCI_Chess960"]
            except KeyError:
                message = "UCI_Chess960 is not supported by the engine " \
                          "and this is a chess960 game."
                logger.critical(message)
                raise RuntimeError(message)

        if game.board().uci_variant != "chess":
            try:
                engine_variants = engine.options["UCI_Variant"].var
                if not game.board().uci_variant in engine_variants:
                    raise AssertionError
            except KeyError:
                message = "UCI_Variant option is not supported by the " \
                          "engine and this is a variant game."
                logger.critical(message)
                raise RuntimeError(message)
            except AssertionError:
                message = "Variant {} is not supported by the engine.".format(
                    game.board().uci_variant)
                logger.critical(message)
                raise RuntimeError(message)

        # Now that engine support for the variant is confirmed, set engine UCI
        # options as appropriate for the variant
        engine.setoption({
            "UCI_Variant": game.board().uci_variant,
            "UCI_Chess960": game.board().chess960,
            "Threads": threads
        })
    else:
        engine.setoption({
            "Threads": threads
        })

    ###########################################################################
    # Clear existing comments and variations
    ###########################################################################
    game = clean_game(game)

    annotation_side = "both"
    if game.headers["White"] == USER:
        annotation_side = "white"
    elif game.headers["Black"] == USER:
        annotation_side = "black"

    ###########################################################################
    # Attempt to classify the opening and calculate the game length
    ###########################################################################
    game, root_node, ply_count = classify_opening(game)
    # root_node starts from the end of the opening, not the beginning of the game

    ###########################################################################
    # Perform game analysis
    ###########################################################################

    logger.debug("depth is %i seconds", arg_depth)

    # Loop through the game doing shallow analysis
    logger.info("Performing first pass...")

    node = game.end()

    white_cpl = []
    black_cpl = []

    while not node == root_node:

        prev_node = node.parent

        # node.board().turn == true -> white to play
        # node.board().turn == false -> black to play

        side_to_play = "white" if prev_node.board().turn else "black"
        if annotation_side != "both" and annotation_side != side_to_play:
            node = prev_node
            continue

        # Get the engine judgment of the played move in this position
        judgment = judge_move(prev_node.board(), node.move, engine,
                              info_handler, arg_depth)

        node.comment = judgment

        # Update CPL list
        delta = judgment["besteval"] - judgment["playedeval"]
        if node.board().turn:
            black_cpl.append(cpl(delta))
        else:
            white_cpl.append(cpl(delta))

        # Create final annotations
        if needs_annotation(judgment):
            add_annotation(node, judgment)
        else:
            node.comment = judgment["playedcomment"]

        # Print some debugging info
        debug_print(node, judgment)

        node = prev_node

    # Calculate the average centipawn loss (ACPL) for each player
    if annotation_side != "black":  # white or both
        node.root().headers["WhiteACPL"] = str(round(acpl(white_cpl)))
    if annotation_side != "white":  # black or both
        node.root().headers["BlackACPL"] = str(round(acpl(black_cpl)))

    ###########################################################################

    annotator = engine.name if engine.name else ""
    node.root().comment = annotator
    node.root().headers["Annotator"] = annotator

    return node.root()


def checkgame(game):
    """
    Check for PGN parsing errors and abort if any were found
    This prevents us from burning up CPU time on nonsense positions
    """
    if game.errors:
        errormsg = "There were errors parsing the PGN game:"
        logger.critical(errormsg)
        for error in game.errors:
            logger.critical(error)
        logger.critical("Aborting...")
        raise RuntimeError(errormsg)

    # Try to verify that the PGN file was readable
    if game.end().parent is None:
        errormsg = "Could not render the board. Is the file legal PGN?" \
                   "Aborting..."
        logger.critical(errormsg)
        raise RuntimeError(errormsg)


def main():
    """
    Main function

    - Load games from the PGN file
    - Annotate each game, and print the game with the annotations
    """
    args = parse_args()
    setup_logging(args)
    engine = args.engine.split()

    pgnfile = args.file

    if pgnfile is None:
        pgnfile = os.path.join(os.getcwd(), "lichess_game.pgn")

    # download if the file is absent
    if not os.path.isfile(pgnfile):
        pgn = lichess.api.user_games(USER, max=args.numgames, format=SINGLE_PGN)
        with open(pgnfile, 'w+') as f:
            f.write(pgn)

    pgn_out_file = args.pgnoutfile
    if pgn_out_file is None:
        pgn_out_file = os.path.join(os.getcwd(), "lichess_game_out.pgn")
    print(pgn_out_file)

    if not args.output_append and os.path.isfile(pgn_out_file):
        os.remove(pgn_out_file)

    try:
        with open(pgnfile) as pgn:
            for game in iter(lambda: chess.pgn.read_game(pgn), None):
                try:
                    analyzed_game = analyze_game(game, args.depth,
                                                 engine, args.threads)
                except KeyboardInterrupt:
                    logger.critical("\nReceived KeyboardInterrupt.")
                    raise
                except Exception as e:
                    logger.critical("\nAn unhandled exception occurred: {}"
                                    .format(type(e)))
                    raise e
                else:
                    with open(pgn_out_file, "a") as f:
                        f.write(str(analyzed_game) + "\n\n")

    except PermissionError:
        errormsg = "Input file not readable. Aborting..."
        logger.critical(errormsg)
        raise


if __name__ == "__main__":
    os.chdir("/Users/kishensurajp/general/chess/python_annotator/")
    main()

# Bash function
# function liano(){
#     current_time=$(date "+%Y.%m.%d-%H.%M.%S")
#     input_pgn="/Users/kishensurajp/general/chess/python_annotator/file_${current_time}.pgn"
#     out_pgn="/Users/kishensurajp/general/chess/python_annotator/file_out_${current_time}.pgn"
#     /Users/kishensurajp/general/chess/python_annotator/python-chess-annotator/annotator/annotate_by_depth.py -f "${input_pgn}" -o "${out_pgn}" -n $1 -d 15
#     /Applications/ScidvsMac.app/Contents/MacOS/scid ~/general/chess/scid/my_games_imp/my_games_imp.si4 "${out_pgn}"
# }
