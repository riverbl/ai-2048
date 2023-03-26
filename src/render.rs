use std::{
    io::{self, Write},
    mem::MaybeUninit,
    os::fd::AsRawFd,
};

const SQUARE_HEIGHT: u32 = 3;
// const SQUARE_WIDTH: u32 = 2 * SQUARE_HEIGHT | 2;
const TOP_ROW: &[u8] = "┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓\n".as_bytes();
const SEPERATOR_ROW: &[u8] = "┣━━━━━━━━╋━━━━━━━━╋━━━━━━━━╋━━━━━━━━┫\n".as_bytes();
const BOTTOM_ROW: &[u8] = "┗━━━━━━━━┻━━━━━━━━┻━━━━━━━━┻━━━━━━━━┛\n".as_bytes();
const EMPTY_ROW: &[u8] = "┃        ┃        ┃        ┃        ┃\n".as_bytes();
const EMPTY_CELL: &[u8] = "┃        ".as_bytes();

fn draw_board_row(out: &mut impl Write, row: u16) -> io::Result<()> {
    for j in 0..4 {
        let exponent = (row >> (j * 4)) & 0xf;

        if exponent == 0 {
            out.write_all(EMPTY_CELL)?;
        } else {
            let n = 1 << exponent;

            write!(out, "┃{n:^8}")?;
        }
    }

    Ok(())
}

pub fn draw_board(out: &mut impl Write, board: u64, score: u32) -> io::Result<()> {
    write!(out, "\nScore: {score}\n")?;
    out.write_all(TOP_ROW)?;

    for i in 0..4 {
        if i != 0 {
            out.write_all(SEPERATOR_ROW)?;
        }

        for _ in 0..((SQUARE_HEIGHT - 1).div_floor(2)) {
            out.write_all(EMPTY_ROW)?;
        }

        draw_board_row(out, (board >> (i * 16)) as u16)?;
        out.write_all("┃\n".as_bytes())?;

        for _ in 0..((SQUARE_HEIGHT - 1).div_ceil(2)) {
            out.write_all(EMPTY_ROW)?;
        }
    }

    out.write_all(BOTTOM_ROW)?;

    Ok(())
}

pub fn redraw_board(
    out: &mut impl Write,
    old_board: u64,
    new_board: u64,
    old_score: u32,
    new_score: u32,
) -> io::Result<()> {
    let mut current_line = if new_score != old_score {
        let target_line = SQUARE_HEIGHT * 4 + 6;
        write!(out, "\x1b[{target_line}FScore: {new_score}")?;

        target_line
    } else {
        0
    };

    let changed_rows =
        (0..4).filter(|i| (new_board >> (i * 16)) & 0xffff != (old_board >> (i * 16)) & 0xffff);

    for row in changed_rows {
        let final_row_to_end = (SQUARE_HEIGHT - 1).div_ceil(2) + 2;
        let between_rows = SQUARE_HEIGHT + 1;
        let target_line = final_row_to_end + between_rows * (3 - row);

        if target_line > current_line {
            write!(out, "\x1b[{}F", target_line - current_line)?;
        } else {
            write!(out, "\x1b[{}E", current_line - target_line)?;
        }

        current_line = target_line;
        draw_board_row(out, (new_board >> (row * 16)) as u16)?;
    }

    if current_line != 0 {
        write!(out, "\x1b[{current_line}E")?;
        out.flush()?;
    }

    Ok(())
}

pub fn setup_terminal(fd: &impl AsRawFd) -> io::Result<()> {
    let fd = fd.as_raw_fd();
    let mut termios = MaybeUninit::uninit();

    let mut termios = unsafe {
        if libc::tcgetattr(fd, termios.as_mut_ptr()) != 0 {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "Error calling tcgetattr",
            ))?;
        }

        termios.assume_init()
    };

    // let c_oflag = termios.c_oflag;

    // unsafe {
    //     libc::cfmakeraw(&mut termios);
    // }

    termios.c_lflag &= !(libc::ECHO | libc::ICANON);

    unsafe {
        if libc::tcsetattr(fd, libc::TCSADRAIN, &termios) != 0 {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "Error calling tcsetattr",
            ))?;
        }
    }

    Ok(())
}
