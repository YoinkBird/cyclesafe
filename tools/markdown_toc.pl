#!/usr/bin/perl




sub basename{
  return((split('/',shift))[-1])
}
my $scriptName = basename($0);
my $fileName = $ARGV[0];

# mode?
my $mode = "mini_toc";
if($ARGV[1]){
  if($ARGV[1] eq "bc_toc"){
    # breadcrumb toc
    $mode = "bc_toc";
  }
  elsif($ARGV[1]){
    # breadcrumb toc
    $mode = $ARGV[1];
  }
}

#### Process incoming text: ###########################
my $text;
{
        local $/;               # Slurp the whole file
        $text = <>;
}

# IDEA: mini TOC . get line-number for <!--toc_mini--> comments. build parse-tree/hash of line-num: headers '####'. insert bulletted-list of headers below line-number for <!--toc_mini-->
#
my %header_hash;
#while (<>){
@text = split("\n",$text);
my $commentBegin = 0;
my $commentEnd   = 0;
for(my $lineNo=0; $lineNo < scalar(@text); $lineNo++){
  my $line = $text[$lineNo];
  # skip comments. simple match, i.e. will any line with a comment anywhere on it
  #+ however, markdown doesn't allow this either:
  #+ This markdown ...
  #+ |## Data Sources <!--
  #+ |-->
  #+ ... renders in browser as:
  #+ |Data Sources <!--
  #+ |
  #+ |-->
  if($line =~ m/<!--/){
    $commentBegin = 1;
  }
  if($line =~ m/-->/){
    $commentBegin = 0;
    $commentEnd = 1;
  }
  # remove any comments, e.g. if comment on one line:  |# Future Work <!-- -->
  $line =~ s/<!--.*//;
  next if($commentBegin);
  # match headers
  if($line =~ m{
      ^(\#{1,6})  # $1 = string of #'s
      [ \t]*
      (.+?)    # $2 = Header text
      [ \t]*
      \#*      # optional closing #'s (not counted)
      $
    }gmx){
    #print($line . "\n");
    my $infHref = {'text' => $2, 'level' => length($1)};
    $header_hash{$lineNo} = $infHref;
  }
}

sub get_mini_toc{
# print(join("\n",sort({ $a <=> $b } keys(%header_hash))) . "\n");
  my $startLine = shift; #2; #testing
  my $startLevel = 0; #testing
  my $started = 0; #testing
  my $prevHighest=0;
  my $prevLine=0;
  my @ul_arr;
  for my $ln (sort({ $a <=> $b } keys(%header_hash))){
    printf("%s %s %s\n", 'ln','lvl','txt') if(0);
    printf("%s %s %s\n",
      $ln,
      $header_hash{$ln}{'level'},
      $header_hash{$ln}{'text'},
    ) if(0);
    if ($started > 1){
      next;
    }
    my $cur_lev = $header_hash{$ln}{'level'};
    next if($ln < $startLine); # ignore all previous
    if($ln >= $startLine && $started == 0){
      $startLevel = $header_hash{$ln}{'level'};
      $started = 1;
    }
    # quit if already started building ul and current level receeds beyond initial level
    if ($cur_lev < $startLevel){
      if($started == 1){
        $started++;
      }
      next;
    }

    if($cur_lev < $prevHighest){
      print("new tree\n") if(0);
    }
    elsif($cur_lev == $prevHighest){
      print("parallel tree\n") if(0);
    }
    $prevHighest = $cur_lev;
    # print($ul_li . "\n";
    # vvv don't need leading '*' vvv
    # my $ul_li = sprintf('* ' x ($cur_lev - $startLevel + 1) . $header_hash{$ln}{'text'});
    # vvv indent by level-num spaces, then add '* ' vvv
    my $ul_li = sprintf('  ' x ($cur_lev - $startLevel + 0) . '* ' . $header_hash{$ln}{'text'});
    # don't include headers 4th and below
    my $cutoff_limit = 4;
    push(@ul_arr, $ul_li) if ($cur_lev < $cutoff_limit) ;

  }
  return @ul_arr;
}

sub get_header_anchortext{
# print(join("\n",sort({ $a <=> $b } keys(%header_hash))) . "\n");
  my $cutoff_limit = shift;
  # vvv testing vvvv
  # only use if want to focus on certain indentation level
  my $startLine = shift if(@_); #testing
  my $startLevel = 0; #testing
  my $started = 0; #testing
  my $prevHighest=0;
  my $prevLine=0;
  my @ul_arr;
  my @bc_arr;
  for my $ln (sort({ $a <=> $b } keys(%header_hash))){
    printf("%s %s %s\n", 'ln','lvl','txt') if(0);
    printf("%s %s %s\n",
      $ln,
      $header_hash{$ln}{'level'},
      $header_hash{$ln}{'text'},
    ) if(0);
    if ($started > 1){
      next;
    }
    my $cur_lev = $header_hash{$ln}{'level'};
    next if($ln < $startLine); # ignore all previous
    if($ln >= $startLine && $started == 0){
      $startLevel = $header_hash{$ln}{'level'}; # = $cur_lev;
      $started = 1;
    }
    # quit if already started building ul and current level receeds beyond initial level
    if ($cur_lev < $startLevel){
      if($started == 1){
        $started++;
      }
      next;
    }

    if($cur_lev < $prevHighest){
      print("new tree\n") if(0);
    }
    elsif($cur_lev == $prevHighest){
      print("parallel tree\n") if(0);
    }
    $prevHighest = $cur_lev;

    # don't include headers 1th and below
    my $cutoff_limit = 1;

    # TODO:
    # e.g. (cur 4 - start 3 == 1) < 1 -> stop
    # e.g. (cur 4 - start 4 == 0) < 1 -> ok
    # if ( ( $cur_lev - $startLevel ) < $cutoff_limit ) {
    # breadcrumb
    if ( $cur_lev <= $cutoff_limit ){
      # gh markdown header-anchors: 
      # https://gist.github.com/asabaylus/3071099#gistcomment-1593627
      # lowercase
      my $bc_head = lc($header_hash{$ln}{'text'});
      # remove non number, letter, space, hyphen
      $bc_head =~ s/[^0-9a-zA-Z\s-]//g;
      # space to hyphen
      $bc_head =~ s/\s/-/g;
      # uniqify
      # stub
      push(@bc_arr, $bc_head);
    }
  }
  return @bc_arr;
}

sub get_toc_breadcrumb{
  my @toc_l1 = &get_header_anchortext(1); # testing
  my $toc_breadcrumb;
  my $first_anchor = "| ";
  for my $l1 (@toc_l1) {
    my $anchor = sprintf("[%s](#%s)" , $l1,$l1);
    #print($anchor . "\n");
    # exception for TOC
    if($l1 =~ m/table-of-content/){
      $first_anchor .= "$anchor | ";
      next;
    }
    $toc_breadcrumb .= "$anchor | ";
  }
  return($first_anchor . $toc_breadcrumb);
}

sub update_bc_toc{
  my $target = '@breadcrumb';
  # only include h1 in the breadcrumb
  my $fargs = { 'fn' => \&get_toc_breadcrumb, 'args' => ['1'] };
  # add new or update existing
  my $found_flag=0;
  for(my $lineNo=0; $lineNo < scalar(@text); $lineNo++){
    my $line = $text[$lineNo];
    my $toc_ins   = '<!--'   . $target . '-->';
    my $toc_noins = '<!--!'  . $target . '-->';
    my $toc_start = '<!--<'  . $target . '>-->';
    my $toc_end   = '<!--</' . $target . '>-->';
    my $re_insert = qr($toc_ins);
    # add new
    if($line =~ m/$re_insert/){
      # equ to: my $bc_toc = $fp->($h_lev_lim);
      # src: https://stackoverflow.com/a/1235133
      my $bc_toc = $fargs->{fn}->(@{$fargs{'args'}});
      print($toc_noins . "\n");
      print($toc_start . "\n");
      print($bc_toc . "\n");
      print($toc_end   . "\n");
      next;
    }
    my $re_start = qr($toc_start);
    my $re_end = qr($toc_end);
    # clear existing
    # raise
    if($line =~ m/$re_start/){
      $found_flag = 1;
      # lower
    } elsif($line =~ m/$re_end/){
      #my $bc_toc = &get_toc_breadcrumb(1);
      my $bc_toc = $fargs->{fn}->(@{$fargs{'args'}});
      print($toc_start . "\n");
      print($bc_toc . "\n");
      print($toc_end   . "\n");
      $found_flag = 0;
      next;
    }
    if($found_flag == 0){
      print($line . "\n");
    }
  }
}

sub rm_bc_toc{
  my $target = '@breadcrumb';
  # only include h1 in the breadcrumb
  my $fargs = { 'fn' => \&get_toc_breadcrumb, 'args' => ['1'] };
  # add new or update existing
  my $found_flag=0;
  for(my $lineNo=0; $lineNo < scalar(@text); $lineNo++){
    my $line = $text[$lineNo];
    my $toc_ins   = '<!--'   . $target . '-->';
    my $toc_noins = '<!--!'  . $target . '-->';
    my $toc_start = '<!--<'  . $target . '>-->';
    my $toc_end   = '<!--</' . $target . '>-->';
    my $re_insert = qr($toc_ins);
    my $re_noinsert = qr($toc_noins);
    # add new
    if($line =~ m/$re_insert/){
      # equ to: my $bc_toc = $fp->($h_lev_lim);
      # src: https://stackoverflow.com/a/1235133
      my $bc_toc = $fargs->{fn}->(@{$fargs{'args'}});
      if(0){
        print($toc_noins . "\n");
        print($toc_start . "\n");
        print($bc_toc . "\n");
        print($toc_end   . "\n");
      }
      next;
    }
    # remove
    if($line =~ m/$re_noinsert/){
      if($line =~ m/$re_start/){
        $found_flag=1;
      }
      # equ to: my $bc_toc = $fp->($h_lev_lim);
      # src: https://stackoverflow.com/a/1235133
      my $bc_toc = $fargs->{fn}->(@{$fargs{'args'}});
      print($toc_ins . "\n");
      if(0){
        print($toc_noins . "\n");
        print($toc_start . "\n");
        print($bc_toc . "\n");
        print($toc_end   . "\n");
      }
      next;
    }
    my $re_start = qr($toc_start);
    my $re_end = qr($toc_end);
    # clear existing
    # raise
    if($line =~ m/$re_start/){
      $found_flag = 1;
      # lower
    } elsif($line =~ m/$re_end/){
      #my $bc_toc = &get_toc_breadcrumb(1);
      my $bc_toc = $fargs->{fn}->(@{$fargs{'args'}});
      if(0){
        print($toc_start . "\n");
        print($bc_toc . "\n");
        print($toc_end   . "\n");
      }
      $found_flag = 0;
      next;
    }
    if($found_flag == 0){
      print($line . "\n");
    }
  }
}
# disable by default; can't do both toc breadcrumb and toc_mini because script relies on modifying stdout
#+ would have to run once for toc_mini, then again for toc_breadcrumb on the prevous output
if( $mode eq "bc_toc" ){
  # print(&get_toc_breadcrumb);
  print(&update_bc_toc);
  exit;
}
# disable by default; can't do several modes at once because script relies on modifying stdout
#+ would have to run once for each mode
if( $mode eq "rm_bc_toc" ){
  # print(&get_toc_breadcrumb);
  print(&rm_bc_toc);
  exit;
}
if(0){
  my @mini_toc = &get_mini_toc(0); # testing
  print(join("\n",@mini_toc) . "\n");
  my @mini_toc = &get_mini_toc(3); # testing
  print(join("\n",@mini_toc) . "\n");
  my @mini_toc = &get_mini_toc(2); # testing
  print(join("\n",@mini_toc) . "\n");
  my @mini_toc = &get_mini_toc(8); # testing
  print(join("\n",@mini_toc) . "\n");
}

if( $mode ne "mini_toc" ){
  exit;
}

# add new or update existing
my $found_flag=0;
for(my $lineNo=0; $lineNo < scalar(@text); $lineNo++){
  my $line = $text[$lineNo];
  my $toc_ins   = '<!--toc_mini-->';
  my $toc_noins = '<!--!toc_mini-->';
  my $toc_start = '<!--<toc_mini>-->';
  my $toc_end   = '<!--</toc_mini>-->';
  my $re_insert = qr($toc_ins);
  if($line =~ m/$re_insert/){
    my @mini_toc = &get_mini_toc($lineNo);
    print($toc_noins . "\n");
    print($toc_start . "\n");
    print(join("\n",@mini_toc) . "\n");
    print($toc_end   . "\n");
    next;
  }
  my $re_start = qr($toc_start);
  my $re_end = qr($toc_end);
  # clear existing
  # raise
  if($line =~ m/$re_start/){
    $found_flag = 1;
  # lower
  } elsif($line =~ m/$re_end/){
    my @mini_toc = &get_mini_toc($lineNo);
    print($toc_start . "\n");
    print(join("\n",@mini_toc) . "\n");
    print($toc_end   . "\n");
    $found_flag = 0;
    next;
  }
  if($found_flag == 0){
    print($line . "\n");
  }
}
exit;

__END__

USAGE:

echo outline.md | sh -cuv 'read file; perl ../tools/markdown_toc.pl $file > ${file}_out && mv ${file}_out $file' ; git diff
