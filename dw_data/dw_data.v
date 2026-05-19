`timescale 1ns/1ps

module tb;

    // ── DW_exp2 ───────────────────────────────────────────────────────
    reg  [15:0] a;
    wire [15:0] z;
    reg signed [7:0] int_signed;
    reg [7:0]  int_mag;
    reg [15:0]  frac_part;
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
    reg  [31:0] b;
    wire [32:0] quotient;

    DW_div #(
        .a_width(33),
        .b_width(32),
        .tc_mode(1'b0),
        .rem_mode(1'b1)
    ) u_dw_div (
        .a(33'h100000000), // Q1.32
        .b(b),             // Q17.15
        .quotient(quotient), // Q16.17
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
        $fdisplay(fd, "# input : i (signed, Q5.12)");
        $fdisplay(fd, "# output: quotient (Q0.16))");

        for (i = -2097152; i < 2097152; i = i + 1) begin
            #10;
            if (i % 100000 == 0)
                $display("Processing i = %0d", i);

            // ── 拆整數和小數部分 ──────────────────────────────────────
            int_signed = i >>> 16;
            frac_part = i[15:0];

            int_mag = (int_signed < 0) ? -int_signed : int_signed;

            if (i > 0 && int_signed > 12) $fdisplay(fd, "%0d %0d", i, 0);
            else if (i < 0 && int_signed <= -12) $fdisplay(fd, "%0d %0d", i, 16'hFFFF);
            else begin

                // ── 送 DW_exp2 ───────────────
                a = frac_part;
                #5;

                // ── 移位得到 2^{±p} ───────────────────────────────────────
                if (i < 0) temp = ({16'b0, z} >> int_mag); // Q17.15
                else       temp = ({16'b0, z} << int_mag); // Q17.15, worst case = 1 << 12 = 13 integer bits

                b = temp + 32'h00008000;
                #10;
                if (quotient[32:17] !== 16'd0) begin
                    $display("[ERROR]: Sigmoid exceeds 1");
                end
                // ── 輸出 Q0.16 ────────────────────────────────────────────
                temp_2 = (quotient[0]) ? (quotient[16:1] + 1) : quotient[16:1];
                $fdisplay(fd, "%0d %0d", i, temp_2);
            end
        end

        $fclose(fd);
        $display("[tb] Done. sig_table.txt written.");
        $finish;
    end

endmodule