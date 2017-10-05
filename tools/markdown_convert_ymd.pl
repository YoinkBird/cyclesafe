#!/usr/bin/perl

sub basename{
  return((split('/',shift))[-1])
}
my $scriptName = basename($0);
my $fileName = $ARGV[0];

#### Process incoming text: ###########################
my $text;
{
        local $/;               # Slurp the whole file
        $text = <>;
}

sub ymd_to_md{
  # yoinkbird-style headers:
  # h1 Header 1
  # h2 Header 2
  # ...
  # No limit, caveat emptor
  $text =~ s{^(h)([\d])\.}{"#" x $2 }egmx;
  return $text;
}

sub md_to_ymd{
  my $text = shift;
  # atx-style headers:
  #  # Header 1
  #  ## Header 2
  #  ## Header 2 with closing hashes ##
  #  ...
  #  ###### Header 6
  #
  $text =~ s{
    ^(\#{1,6})  # $1 = string of #'s
    ([ \t]*)
    #(.+?)    # $2 = Header text
    #[ \t]*
    #*      # optional closing #'s (not counted)
    #\n+
  }{
    my $h_level = length($1);
    "h${h_level}.$2"
  }egmx;
  return $text;
}

if($fileName =~ m/\.md$/){
  print(&md_to_ymd($text));
}
elsif($fileName =~ m/\.ymd$/){
  print(&ymd_to_md($text));
}
if(0){
  if($scriptName =~ m/^md_to_ymd/){
    print(&md_to_ymd($text));
  }
  elsif($scriptName =~ m/^ymd_to_md/){
    print(&ymd_to_md($text));
  }
}

exit 0;
__END__

testing

 echo 'h1.abstract h2.intro' | tr " " "\n" | perl -e 'local $/;$text=(<>); print($text . "\n---\n"); sub expand{($type,$num)=@_; return("#" x $num . " ")}; $line=$_; $text =~ s{^(h)([\d])\.}{"#" x $2 . " "}gmex; print($text)'


cat outline.ymd | perl -ne 'sub expand{($type,$num)=@_; return("#" x $num )}; $line=$_; $line =~ s/^(h)([\d])\./&expand($1,$2)/ge; print $line' > outline.md


echo '#abstract|## intro|furthermore, ### means a header' | tr "|" "\n" | ./md_to_ymd.pl 
h1.abstract
h2. intro
furthermore, ### means a header

