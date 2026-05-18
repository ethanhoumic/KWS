`timescale 1ns/1ps

module tb;

    // ── DW_exp2 ───────────────────────────────────────────────────────
    reg  [15:0] a;
    wire [15:0] z;
    reg signed [7:0] int_signed;
    reg [7:0]  int_mag;
    reg [7:0]  frac_part;
    reg [31:0] temp;
    reg [23:0] temp_2;

    DW_exp2 #(
        .op_width(16),
        .arch(2),
        .err_range(1)
    ) u_dw_exp2 (
        .a(a),
        .z(z)
    );

    // ── DW_div ────────────────────────────────────────────────────────
    reg  [23:0] b;
    wire [32:0] quotient;

    DW_div #(
        .a_width(33),
        .b_width(24),
        .tc_mode(1'b0),
        .rem_mode(1'b1)
    ) u_dw_div (
        .a(33'h100000000),
        .b(b),
        .quotient(quotient), // Q1.32
        .remainder()
    );

    integer  fd;
    integer  i;

    initial begin
        a = 0;
        b = 0;
        int_signed = 0;
        int_mag = 0;
        frac_part = 0;
        temp = 0;
        temp_2 = 0;

        fd = $fopen("sig_table.txt", "w");
        if (fd == 0) begin
            $display("ERROR: cannot open sig_table.txt");
            $finish;
        end

        $fdisplay(fd, "# sigmoid lookup table");
        $fdisplay(fd, "# input : i (signed, Q5.8)");
        $fdisplay(fd, "# output: quotient (Q0.16))");

        for (i = -8192; i < 8192; i = i + 1) begin
            #10;
            if (i % 500 == 0)
                $display("Processing i = %0d", i);

            // ── 拆整數和小數部分 ──────────────────────────────────────
            int_signed = i >>> 8;
            frac_part = i[7:0];

            if (i < 0 && frac_part != 0) begin
                int_signed = int_signed - 1;
                frac_part = 12'h100 - frac_part;
            end

            int_mag = (int_signed < 0) ? -int_signed : int_signed;

            // ── 送 DW_exp2：frac 左移 8-bit 對齊 Q0.16 ───────────────
            a = {frac_part, 8'b0};
            #5;

            // ── 移位得到 2^{±p} ───────────────────────────────────────
            if (i < 0) temp = ({16'b0, z} >> int_mag); // Q17.15
            else       temp = ({16'b0, z} << int_mag); // Q17.15, worst case = 1 << 12 = 13 integer bits

            // truncate to Q15.8

            temp_2 = (temp[6]) ? (temp[29:7] + 1) : temp[29:7];

            // ── 除法器：b = Q16.8 ──────────────────────────────
            b = temp_2 + 24'h000100;
            #10;
            if (quotient[32] === 1'b1) begin
                $display("[ERROR]: Sigmoid exceeds 1");
            end
            // ── 輸出 Q0.32 ────────────────────────────────────────────
            if (i > 0 && int_signed > 12) $fdisplay(fd, "%0d %0d", i, 0);
            else if (i < 0 && int_signed <= -12) $fdisplay(fd, "%0d %0d", i, 32'hFFFFFFFF);
            else $fdisplay(fd, "%0d %0d", i, quotient[31:0]);
        end

        $fclose(fd);
        $display("[tb] Done. sig_table.txt written.");
        $finish;
    end

endmodule