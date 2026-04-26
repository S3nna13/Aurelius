"""Generate math_generator.py from compact template definitions."""
import math
from pathlib import Path

T = []  # template collector

def M(name, templates):
    """Register a method with name and {diff: [(compute_code, inst_code, resp_code)]}"""
    T.append((name, templates))

# ========= HELPER =========
def mk(compute, inst, resp):
    return (compute, inst, resp)

# ========= ARITHMETIC =========
M("arithmetic", {
    "easy": [
        mk("a,b=__(10,999),__(10,999); ans=a+b",
           '"Compute {}+{}."',
           '"Step 1: Add units.\\nStep 2: Add tens and hundreds.\\nTherefore, {0}."'),
        mk("a,b=__(10,999),__(10,999);\nif a<b: a,b=b,a\nans=a-b",
           '"Compute {}-{}."',
           '"Step 1: {}-{}={}.\\nTherefore, {}."'),
        mk("a,b=__(2,12),__(2,12); ans=a*b",
           '"Compute {}x{}."',
           '"Step 1: {}x{}={}.\\nTherefore, {}."'),
        mk("a,b=__(2,10),__(2,10); prod=a*b",
           '"Divide {} by {}."',
           '"Step 1: {}x{}={}.\\nStep 2: {}/{}={}.\\nTherefore, {}."'),
        mk("a,b,c=__(2,10),__(2,10),__(2,10); ans=a+b*c",
           '"Compute {}+{}x{}."',
           '"Step 1: {}x{}={}.\\nStep 2: {}+{}={}.\\nTherefore, {}."'),
        mk("a,b=__(1,30),__(1,30); s=a+b; avg=s/2",
           '"Average of {} and {}?"',
           '"Step 1: Sum={}.\\nStep 2: {}/2={}.\\nTherefore, {}."'),
        mk("a=__(1,15); ans=a*a",
           '"Compute {}^2."',
           '"Step 1: {}^2={}x{}={}.\\nTherefore, {}."'),
        mk("a=__(1,10); ans=2**a",
           '"Compute 2^{}."',
           '"Step 1: 2x2x...x2({} times)={}.\\nTherefore, {}."'),
        mk("a,b=__(2,20),__(2,10); ans=round(a/b,2)",
           '"Compute {}/{} (2dp)."',
           '"Step 1: {}/{}={:.4f}.\\nStep 2: Rounded: {}.\\nTherefore, {}."'),
        mk("a,b,c=__(2,30),__(2,5),__(1,3); ans=a*b+c",
           '"Compute {}x{}+{}."',
           '"Step 1: {}x{}={}.\\nStep 2: {}+{}={}.\\nTherefore, {}."'),
    ],
    "medium": [
        mk("a,b=__(2,9),__(2,9)\nn=a*__(3,9)\nd=b*__(3,9)\ng=__.gcd(n,d)\nsn,sd=n//g,d//g",
           '"Simplify {}/{}."',
           '"Step 1: GCD={}.\\nStep 2: {} and {} by {}.\\nStep 3: {}/{}={}/{}."'),
        mk("a,b,c,d=__(1,8),__(1,8),__(1,8),__(1,8)\nn=a*d+b*c\ndn=b*d\ng=__.gcd(n,dn)\nsn,sd=n//g,dn//g",
           '"Compute {}/{}+{}/{}."',
           '"Step 1: CD={}.\\nStep 2: {}/{}={}/{}, {}/{}={}/{}.\\nStep 3: Sum={}/{}.\\nStep 4: Simplify: {}/{}."'),
        mk("a,b=__(2,10),__(2,10)\ns=a+b;p=a*b\nv=round(s/p,4)",
           '"Compute 1/{}+1/{}."',
           '"Step 1: CD={}.\\nStep 2: {}/{}+{}/{}={}/{}.\\nStep 3: ={:.4f}.\\nTherefore, {:.4f}."'),
        mk("a=__(2,25);b=__(10,200)\nv=a*b/100\nd=int(v)if v==int(v)else f'{v:.2f}'",
           '"What is {}% of {}?"',
           '"Step 1: {}%={}/100.\\nStep 2: {}/100x{}={:.2f}.\\nTherefore, {}."'),
        mk("a=__(50,200);b=__(10,50)\np=round(b/a*100,1)",
           '"Increase from {} to {}. % increase?"',
           '"Step 1: Increase={}.\\nStep 2: {}/{}={:.4f}.\\nStep 3: x100={}%.\\nTherefore, {}%."'),
        mk("p=__(100,1000);r=__(1,10);t=__(2,5)\nv=round(p*(1+r/100)**t,2)\nf=round((1+r/100)**t,4)",
           '"Compound: ${} at {}% for {}y."',
           '"Step 1: A=P(1+r)^t.\\nStep 2: (1+{})/100)^t={}.\\nStep 3: ${}x{}={:.2f}.\\nTherefore, ${:.2f}."'),
        mk("a=__(2,20);v=__.factorial(a)",
           '"Compute {}!."',
           '"Step 1: {}!={}.\\nTherefore, {}."'),
        mk("a,b=__(1,50),__(1,50)\nif a>b:a,b=b,a\nq=b//a;r=b%a",
           '"Remainder: {}/{}?"',
           '"Step 1: {}={}x{}+{}.\\nTherefore, r={}."'),
        mk("a,b,c=__(1,9),__(1,9),__(1,5);v=a*b+c*c",
           '"Compute {}x{}+{}^2."',
           '"Step 1: {}^2={}.\\nStep 2: {}x{}={}.\\nStep 3: Sum={}.\\nTherefore, {}."'),
        mk("a,b=__(1,9),__(1,9)\nc=__(1,9);v=a*b-c",
           '"Compute {}x{}-{}."',
           '"Step 1: {}x{}={}.\\nStep 2: {}-{}={}.\\nTherefore, {}."'),
    ],
    "hard": [
        mk("a=__(2,6);b=__(2,6)\nn=a+b;k=a\nv=__.comb(n,k)",
           '"Compute C({},{})"',
           '"Step 1: C({},{})={}!/({}!({}-{})!).\\nStep 2: {}!={}.\\nStep 3: {}!={}.\\nStep 4: {}!={}.\\nStep 5: ={}.\\nTherefore, {}."'),
        mk("a=__(2,200)\nr=int(__.isqrt(a))\nif r*r==a:a+=1;r=int(__.isqrt(a))",
           '"Rational? sqrt({})"',
           '"Step 1: {}^2={}, {}^2={}.\\nStep 2: {} not perfect square.\\nTherefore, irrational."'),
        mk("a=__(2,10)\nv=a*(a+1)*(2*a+1)//6",
           '"Sum 1^2..{}^2?"',
           '"Step 1: n(n+1)(2n+1)/6.\\nStep 2: {}({})({})/6.\\nStep 3: {}x{}={}.\\nStep 4: x{}={}.\\nStep 5: /6={}.\\nTherefore, {}."'),
        mk("a,b,c,d=__(2,9),__(2,9),__(2,9),__(2,9)\nn=a*d\ndn=b*c\ng=__.gcd(n,dn)\nsn,sd=n//g,dn//g",
           '"Compute ({}/{})/({}/{})"',
           '"Step 1: Flip: ({}/{})x({}/{}).\\nStep 2: ({}x{})/({}x{})={}/{}.\\nStep 3: GCD={}.\\nStep 4: Simplify={}/{}.\\nTherefore, {}."'),
        mk("a,b=__(2,12),__(2,12)\nl=a*b//__.gcd(a,b)\ng=__.gcd(a,b)",
           '"LCM of {} and {}?"',
           '"Step 1: LCM=axb/GCD.\\nStep 2: GCD={}.\\nStep 3: {}x{}={}.\\nStep 4: {}/{}={}.\\nTherefore, LCM={}."'),
        mk("a,b=__(2,8),__(2,8)\nv=(a+b)**2",
           '"Expand ({}+{})^2."',
           '"Step 1: (x+y)^2=x^2+2xy+y^2.\\nStep 2: {}^2={}.\\nStep 3: 2x{}x{}={}.\\nStep 4: {}^2={}.\\nStep 5: Sum={}.\\nTherefore, {}."'),
        mk("a,b=__(2,30),__(2,30)\nx,y=a,b;st=[]\nwhile y:r=x%y;st.append(f\"{x}={y}x{x//y}+{r}\" if r else f\"{x}={y}x{x//y}\");x,y=y,r",
           '"GCD({},{}) via Euclid?"',
           '"Step 1: Euclidean algorithm.\\nStep 2: Last non-zero={}.\\nTherefore, GCD={}."'),
        mk("a,b=__(1,12),__(1,12)\ncs=a*a+b*b\nc=round(__.sqrt(cs),2)",
           '"sqrt({}^2+{}^2)?"',
           '"Step 1: {}^2={}.\\nStep 2: {}^2={}.\\nStep 3: Sum={}.\\nStep 4: sqrt={}.\\nTherefore, {}."'),
        mk("a,b,c=__(2,6),__(1,5),__(1,5);v=a**b+a**c",
           '"Compute {}^{}+{}^{}"',
           '"Step 1: {}^{}={}.\\nStep 2: {}^{}={}.\\nStep 3: Sum={}.\\nTherefore, {}."'),
        mk("a,b=__(1,6),__(1,6)\nc=__(1,6)\nv=__.factorial(a)//(__.factorial(b)*__.factorial(c))",
           '"Compute {}!/({}!x{}!)"',
           '"Step 1: {}!={}.\\nStep 2: {}!={}.\\nStep 3: {}!={}.\\nStep 4: denom={}x{}={}.\\nStep 5: {}/{}={}.\\nTherefore, {}."'),
    ],
})

# ========= ALGEBRA =========
M("algebra", {
    "easy": [
        mk("a,b=__(1,10),__(1,20);x=__(1,10);r=a*x+b",
           '"Solve {}x+{}={}"',
           '"Step 1: -{}: {}x={}.\\nStep 2: /{}: x={}.\\nTherefore, x={}."'),
        mk("a,b=__(1,10),__(1,10);x=__(1,10)\nif x==0:x=3\nr=a*x-b",
           '"Solve {}x-{}={}"',
           '"Step 1: +{}: {}x={}.\\nStep 2: /{}:x={}.\\nTherefore, x={}."'),
        mk("a,b,c=__(1,5),__(1,5),__(1,10);x=__(1,10);r=a*x+b*x+c;t=a+b",
           '"Solve {}x+{}x+{}={}"',
           '"Step 1: Combine: ({}x+{})x+{}={}x+{}.\\nStep 2: -{}: {}x={}.\\nStep 3: /{}: x={}.\\nTherefore, x={}."'),
        mk("a,b=__(1,5),__(1,5);x=__(2,10);r=a*(x+b)",
           '"Solve {}(x+{})={}"',
           '"Step 1: /{}: x+{}={}.\\nStep 2: -{}: x={}.\\nTherefore, x={}."'),
        mk("a=__(1,5);x=__(1,10);r=a*(x-3)+5",
           '"Solve {}(x-3)+5={}"',
           '"Step 1: Expand: {}x-{}+5={}.\\nStep 2: Simplify: {}x-{}={}.\\nStep 3: +{}: {}x={}.\\nStep 4: /{}: x={}.\\nTherefore, x={}."'),
        mk("a,k=__(1,5),__(2,10);x=__(1,10);r=a/k*x",
           '"Solve ({}/{})x={}"',
           '"Step 1: x{}: {}x={}.\\nStep 2: /{}: x={}.\\nTherefore, x={}."'),
        mk("m,b=__(1,5),__(1,10);x=__(1,10);y=m*x+b",
           '"If y={}x+{}, find y when x={}."',
           '"Step 1: y={}({})+{}.\\nStep 2: y={}+{}={}.\\nTherefore, y={}."'),
        mk("a=__(1,9);x=__(1,10);r=a*x",
           '"Solve {}x={}"',
           '"Step 1: /{}: x={}.\\nTherefore, x={}."'),
        mk("a,b=__(2,5),__(2,5);x=__(1,10);r=a*x+b*x;t=a+b",
           '"Solve {}x+{}x={}"',
           '"Step 1: Combine: {}x.\\nStep 2: x={}/{}={}.\\nTherefore, x={}."'),
        mk("a,b,c=__(1,5),__(1,5),__(1,10);x=__(1,5);t=a+b;r=a*x+b*x-c",
           '"Solve {}x+{}x-{}={}"',
           '"Step 1: Combine: {}x-{}={}.\\nStep 2: +{}: {}x={}.\\nStep 3: /{}: x={}.\\nTherefore, x={}."'),
    ],
    "medium": [
        mk("a,b=__(1,5),__(1,10);x=__(1,5);c=a*x+b",
           '"Solve {}x+{}={}"',
           '"Step 1: -{}: {}x={}.\\nStep 2: /{}: x={}.\\nVerify: {}x{}+{}={}.\\nTherefore, x={}."'),
        mk("a,b,c=__(1,5),__(1,5),__(1,5);x=__(1,5);r=a*x+c",
           '"Solve {}x={}x+{}"',
           '"Step 1: -{}x: ({}x-{})x={}.\\nStep 2: {}x={}.\\nStep 3: /{}: x={}.\\nVerify: {}x{}={}, {}x{}+{}={}.\\nTherefore, x={}."'),
        mk("a,b,c,d=__(1,4),__(1,4),__(1,4),__(1,4);x=__(1,5)\nr=a*x+b;l=c*x-d",
           '"Solve {}x-{}={}x+{}"',
           '"Step 1: -{}x: ({}x-{})x-{}={}.\\nStep 2: +{}: ({}x-{})x={}.\\nStep 3: /{}: x={}.\\nTherefore, x={}."'),
        mk("a,b=__(1,5),__(1,10);r=__(1,5)\nsq=r*r;sol1=b+r;sol2=b-r",
           '"Solve (x-{})^2={}"',
           '"Step 1: sqrt: x-{}={}.\\nStep 2: x={}{}.\\nStep 3: x={} or {}.\\nTherefore, x={} or {}."'),
        mk("a,b=__(1,5),__(1,10);x=__(1,10);y=a*x+b\nsl=(y-b)/x;sl=int(sl)if sl==int(sl)else sl",
           '"Slope through (0,{}) and ({},{})?"',
           '"Step 1: m=({}-{})/({}-0)={}.\\nStep 2: y-int={}.\\nTherefore, m={}, b={}."'),
        mk("a,k=__(1,3),__(1,5);y=__(1,5)\nrat=a*k/k;ny=rat*(k+2)",
           '"y varies with x. y={} when x={}. y when x={}?"',
           '"Step 1: k=y/x={}/{}.\\nStep 2: y=kx. y={}({})={}.\\nTherefore, y={}."'),
        mk("a,b=__(1,4),__(1,5);x=__(1,10);v=a*x+b",
           '"Evaluate {}x+{} when x={}."',
           '"Step 1: {}({})+{}={}+{}={}.\\nTherefore, {}."'),
        mk("a,b=__(1,3),__(1,5);x=__(1,10)",
           '"Expand ({}x+{})(x-1)"',
           '"Step 1: FOIL: {}x^2+({}x)(-1)+{}(x)+(b)(-1).\\nStep 2: {}x^2+{}x+{}.\\nTherefore, {}x^2+{}x-{}."'),
        mk("a,b=__(1,4),__(1,5);x=__(1,10);v=2*x+a",
           '"Find x if 2x+{}={}"',
           '"Step 1: -{}: 2x={}.\\nStep 2: /2: x={}.\\nTherefore, x={}."'),
        mk("a,b=__(1,4),__(1,5);r=b",
           '"Solve (2x)/{}={}"',
           '"Step 1: x{}: 2x={}x{}={}.\\nStep 2: /2: x={}/2={}.\\nTherefore, x={}."'),
    ],
    "hard": [
        mk("r1=__(-5,5);r2=__(-5,5)\nwhile r2==r1 or r1==0 or r2==0:r2=__(-5,5)\nb=-(r1+r2);c=r1*r2",
           '"Find roots of x^2+{}x+{}=0"',
           '"Step 1: Find numbers with product {} and sum {}.\\nStep 2: {} and {}.\\nStep 3: (x{:+d})(x{:+d})=0.\\nStep 4: x={} or {}.\\nTherefore, x={} or {}."'),
        mk("x1,y1=__(0,5),__(1,10);x2,y2=__(6,10),__(1,10)\nsl=(y2-y1)/(x2-x1);sl=int(sl)if sl==int(sl)else sl\nic=y1-sl*x1;ic=int(ic)if ic==int(ic)else ic",
           '"Line through ({},{}) and ({},{})?"',
           '"Step 1: m=({}-{})/({}-{})={}.\\nStep 2: y-{}={}(x-{}).\\nStep 3: y={}x+{}.\\nTherefore, y={}x+{}."'),
        mk("a,b,c=__(1,3),__(1,3),__(1,10)\nx,y,z=__(1,5),__(1,5),__(1,5)\ne1=a*x+b*y+c*z\ne2=a*x-b*y+c*z\ne3=a*x+b*y-c*z",
           '"System:\\n{}x+{}y+{}z={}\\n{}x-{}y+{}z={}\\n{}x+{}y-{}z={}"',
           '"Step 1: Eq1-Eq2: {}y={} => y={}.\\nStep 2: Eq1-Eq3: {}z={} => z={}.\\nStep 3: Back-sub: x={}, y={}, z={}.\\nTherefore, x={}, y={}, z={}."'),
        mk("a=__(1,5);x=__(1,5)\nr=abs(a*x-10)\ns1=(r+10)/a;s2=(-r+10)/a\ns1=int(s1)if s1==int(s1)else s1\ns2=int(s2)if s2==int(s2)else s2",
           '"Solve |{}x-10|={}"',
           '"Step 1: {}x-10={} or {}x-10=-{}.\\nStep 2: Case1: {}x={} => x={}.\\nStep 3: Case2: {}x={} => x={}.\\nTherefore, x={} or {}."'),
        mk("a,b=__(1,5),__(1,5);x=__(2,8)\nn=a*x+b;dn=x-b;v=n/dn\nv=int(v)if v==int(v)else v",
           '"Evaluate ({}x+{})/(x-{}) when x={}"',
           '"Step 1: Num={}({})+{}={}.\\nStep 2: Den={}-{}={}.\\nStep 3: {}/{}={}.\\nTherefore, {}."'),
        mk("a=__(1,5);x=__(1,5);r=a*x**3",
           '"Solve {}x^3={}"',
           '"Step 1: /{}: x^3={}.\\nStep 2: cbrt({})={}.\\nTherefore, x={}."'),
        mk("a,b=__(1,3),__(1,3);x=__(1,5)\np=a*b;d=(a+b)**2-4*a*b",
           '"Discriminant of x^2-{}x+{}=0?"',
           '"Step 1: D=b^2-4ac.\\nStep 2: a=1,b={},c={}.\\nStep 3: D={}^2-4x{}x{}={}.\\nTherefore, D={}."'),
        mk("a,b=__(1,4),__(1,4);x,y=__(1,5),__(1,5)\ne1=a*x+b*y\ne2=a*x-b*y\nxv=(e1+e2)//(2*a);yv=(e1-e2)//(2*b)",
           '"Solve:\\n{}x+{}y={}\\n{}x-{}y={}"',
           '"Step 1: Add: {}x={} => x={}.\\nStep 2: Sub: {}x{}+{}y={}.\\nStep 3: {}y={} => y={}.\\nTherefore, x={}, y={}."'),
        mk("a,b=__(2,5),__(2,5)",
           '"Inverse of f(x)={}x+{}?"',
           '"Step 1: y={}x+{}.\\nStep 2: Swap: x={}y+{}.\\nStep 3: {}y=x-{} => y=(x-{})/{}.\\nTherefore, f^-1(x)=(x-{})/{}."'),
        mk("a=__(1,5);x=__(1,10);r=a*x+3*x;t=a+3;xv=r//t",
           '"Solve {}x+3x={}"',
           '"Step 1: ({}x+3)x={}x.\\nStep 2: {}x={}.\\nStep 3: x={}.\\nTherefore, x={}."'),
    ],
})

# ========= GEOMETRY =========
M("geometry", {
    "easy": [
        mk("w,h=__(3,15),__(3,15);a=w*h",
           '"Area of {}x{} rectangle?"',
           '"Step 1: A=wxh={}x{}={}.\\nTherefore, {}."'),
        mk("w,h=__(3,15),__(3,15);p=2*(w+h)",
           '"Perimeter of {}x{} rectangle?"',
           '"Step 1: P=2({}+{})={}.\\nTherefore, {}."'),
        mk("r=__(1,10);a=round(__.pi*r*r,2)",
           '"Area of circle r={}?"',
           '"Step 1: A=pi x r^2=pi x {} = {}.\\nTherefore, {}."'),
        mk("b,h=__(3,15),__(3,15);a=0.5*b*h;a=int(a)if a==int(a)else a",
           '"Area of triangle base={} h={}?"',
           '"Step 1: A=1/2x{}x{}={}.\\nTherefore, {}."'),
        mk("r=__(1,10);c=round(2*__.pi*r,2)",
           '"Circumference r={}?"',
           '"Step 1: C=2*pi*r=2*pi*{}={}.\\nTherefore, {}."'),
        mk("s=__(3,15);a=s*s",
           '"Area of square side={}?"',
           '"Step 1: A=s^2={}^2={}.\\nTherefore, {}."'),
        mk("s=__(3,15);p=4*s",
           '"Perimeter of square side={}?"',
           '"Step 1: P=4s=4x{}={}.\\nTherefore, {}."'),
        mk("a,b=__(3,10),__(3,10);c=round(__.sqrt(a*a+b*b),2)",
           '"Hypotenuse of right triangle legs={},{}?"',
           '"Step 1: c^2={}^2+{}^2.\\nStep 2: c^2={}+{}={}.\\nStep 3: c={}.\\nTherefore, {}."'),
        mk("l,w,h=__(2,8),__(2,8),__(2,8);v=l*w*h",
           '"Volume of {}x{}x{} prism?"',
           '"Step 1: V=lwh={}x{}x{}={}.\\nTherefore, {}."'),
        mk("r,h=__(2,8),__(3,10);v=round(__.pi*r*r*h,2)",
           '"Volume of cylinder r={} h={}?"',
           '"Step 1: V=pi x r^2 x h = pi x {} x {} = {}.\\nTherefore, {}."'),
    ],
    "medium": [
        mk("r=__(3,10);v=round(4/3*__.pi*r**3,2)",
           '"Sphere volume r={}?"',
           '"Step 1: V=4/3*pi*r^3=4/3*pi*{}={}.\\nTherefore, {}."'),
        mk("a,b=__(3,10),__(3,10)\narea=int(0.5*a*b)\nc=round(__.sqrt(a*a+b*b),2)",
           '"Right triangle legs {}, {}. Area and hyp?"',
           '"Step 1: Area=1/2x{}x{}={}.\\nStep 2: c^2={}^2+{}^2={}+{}={}.\\nStep 3: c={}.\\nTherefore, area={}, hyp={}."'),
        mk("x1,y1=__(0,10),__(0,10);x2,y2=__(0,10),__(0,10)\nd=round(__.sqrt((x2-x1)**2+(y2-y1)**2),2)",
           '"Distance between ({},{}) and ({},{})?"',
           '"Step 1: d=sqrt(({}-{})^2+({}-{})^2).\\nStep 2: d={}.\\nTherefore, {}."'),
        mk("x1,y1=__(0,10),__(0,10);x2,y2=__(0,10),__(0,10)\nmx=(x1+x2)/2;my=(y1+y2)/2",
           '"Midpoint of ({},{}) and ({},{})?"',
           '"Step 1: (({}+{})/2, ({}+{})/2).\\nStep 2: ({}, {}).\\nTherefore, ({}, {})."'),
        mk("r=__(3,10);sa=round(4*__.pi*r**2,2)",
           '"Sphere surface area r={}?"',
           '"Step 1: SA=4*pi*r^2=4*pi*{}={}.\\nTherefore, {}."'),
        mk("r,h=__(2,8),__(3,10);v=round(1/3*__.pi*r**2*h,2)",
           '"Cone volume r={} h={}?"',
           '"Step 1: V=1/3*pi*r^2*h=1/3*pi*{}*{}={}.\\nTherefore, {}."'),
        mk("a,b,c=__(3,10),__(3,10),__(3,10)\ns=(a+b+c)/2\narea=round(__.sqrt(s*(s-a)*(s-b)*(s-c)),2)",
           '"Heron\'s formula: sides {},{},{}?"',
           '"Step 1: s=({}+{}+{})/2={}.\\nStep 2: A=sqrt({}({}-{})({}-{})({}-{})).\\nStep 3: A={}.\\nTherefore, {}."'),
        mk("a=__(30,70);b=__(30,70);c=180-a-b",
           '"Triangle angles {} and {}. Third?"',
           '"Step 1: A+B+C=180.\\nStep 2: C=180-{}-{}={}.\\nTherefore, {}."'),
        mk("a=__(3,8);a2=round((3*__.sqrt(3)/2)*a*a,2)",
           '"Hexagon area side={}?"',
           '"Step 1: A=3*sqrt(3)/2*a^2.\\nStep 2: A={}.\\nTherefore, {}."'),
        mk("b,h=__(3,12),__(3,12);a=b*h",
           '"Parallelogram area base={} h={}?"',
           '"Step 1: A=b*h={}*{}={}.\\nTherefore, {}."'),
    ],
    "hard": [
        mk("a,b=__(1,5),__(1,5);rs=a*a+b*b;sl=round(-a/b,2)if b else 0",
           '"Tangent slope to x^2+y^2={} at ({},{})?"',
           '"Step 1: 2x+2yy\'=0 => y\'=-x/y.\\nStep 2: y\'=-{}/{}={}.\\nTherefore, {}."'),
        mk("r,h=__(2,8),__(2,8)\nvc=round(__.pi*r*r*h,2)\nvn=round(1/3*__.pi*r*r*h,2)",
           '"Cylinder and cone r={} h={}. Volume ratio?"',
           '"Step 1: V_cyl=pi*{}^2*{}={}.\\nStep 2: V_cone=1/3*pi*{}^2*{}={}.\\nStep 3: Ratio={}:{}=3:1.\\nTherefore, 3:1."'),
        mk("x1,y1=__(0,4),__(0,4);x2,y2=__(5,10),__(5,10)\nx3,y3=__(0,4),__(5,10)\narea=abs(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2\narea=int(area)if area==int(area)else area",
           '"Area of triangle ({},{}), ({},{}), ({},{})?"',
           '"Step 1: Shoelace: A=1/2|x1(y2-y3)+x2(y3-y1)+x3(y1-y2)|.\\nStep 2: A={}.\\nTherefore, {}."'),
        mk("a,b=__(1,5),__(1,5)\nr=round(__.sqrt(a*a+b*b),2)\nt=round(__.degrees(__.atan2(b,a)),2)",
           '"Convert ({},{}) to polar."',
           '"Step 1: r=sqrt({}^2+{}^2)={}.\\nStep 2: theta=arctan({}/{})={}.\\nTherefore, ({}, {})."'),
        mk("", '"Triangle sides 3,4,5. Angle opposite 5?"',
           '"Step 1: Law of Cosines: c^2=a^2+b^2-2ab cos(C).\\nStep 2: cos(C)=({}^2+{}^2-{}^2)/(2x{}x{})=0.\\nStep 3: C=arccos(0)=90.\\nTherefore, 90 degrees."'),
        mk("s=__(3,7);v=s**3;sa=6*s*s;r=round(sa/v,2)",
           '"Cube side {}. SA:V ratio?"',
           '"Step 1: V={}^3={}.\\nStep 2: SA=6x{}^2={}.\\nStep 3: Ratio={}:{}={}:1.\\nTherefore, {}:1."'),
        mk("r=__(3,9);ang=__(30,180)\nal=round(2*__.pi*r*ang/360,2)\nsa=round(__.pi*r*r*ang/360,2)",
           '"Circle r={}, {} deg. Arc length and sector area?"',
           '"Step 1: Arc=2*pi*{}*{}/360={}.\\nStep 2: Sector=pi*{}^2*{}/360={}.\\nTherefore, arc={}, sector={}."'),
        mk("a=__(30,60);rad=round(__.radians(a),4)\nsv=round(__.sin(__.radians(a)),4)\ncv=round(__.cos(__.radians(a)),4)\ntv=round(__.tan(__.radians(a)),4)",
           '"Find sin({}), cos({}), tan({})?"',
           '"Step 1: {} rad = {}.\\nStep 2: sin={}.\\nStep 3: cos={}.\\nStep 4: tan={}.\\nTherefore, sin={}, cos={}, tan={}."'),
        mk("a,b=__(1,8),__(1,8);d=round(__.sqrt(a*a+b*b),2)",
           '"Diagonal of {}x{} rectangle?"',
           '"Step 1: d=sqrt({}^2+{}^2)=sqrt({})={}.\\nTherefore, {}."'),
        mk("a,b=__(1,5),__(1,5)\nrs=a*a+b*b\nsl=round(-a/b,2)if b else 0",
           '"Slope of tangent to x^2+y^2={} at ({},{})?"',
           '"Step 1: y\'=-x/y at ({},{}) = -{}/{}={}.\\nTherefore, {}."'),
    ],
})

# ========= PROBABILITY =========
M("probability", {
    "easy": [
        mk("t=__(4,12);f=__(1,t-1);p=f/t",
           '"Bag: {} balls, {} red. P(red)?"',
           '"Step 1: P=fav/total={}/{}.\\nTherefore, P={}."'),
        mk("", '"Die rolled. P(even)?"',
           '"Step 1: Evens: 2,4,6 (3).\\nStep 2: P=3/6=1/2.\\nTherefore, 1/2."'),
        mk("", '"Die rolled. P(prime)?"',
           '"Step 1: Primes: 2,3,5 (3).\\nStep 2: P=3/6=1/2.\\nTherefore, 1/2."'),
        mk("n=__(2,6)", '"Spinner {} sections. P(1)?"',
           '"Step 1: 1 of {}.\\nStep 2: P=1/{}.\\nTherefore, 1/{}."'),
        mk("t=__(10,30);f=__(1,5);p=f/t",
           '"Class {} students, {} glasses. P(glasses)?"',
           '"Step 1: P={}/{}.\\nTherefore, P={}."'),
        mk("f=__(1,5);t=__(6,12);p=1-f/t",
           '"P(rain)={}/{}. P(no rain)?"',
           '"Step 1: P=1-{}/{}={}/{}.\\nTherefore, {}."'),
        mk("", '"Two coins. P(one head)?"',
           '"Step 1: Outcomes: HH,HT,TH,TT.\\nStep 2: Favorable: HT,TH (2).\\nStep 3: P=2/4=1/2.\\nTherefore, 1/2."'),
        mk("n=__(2,10)", '"Number 1..{}. P(multiple of 1)?"',
           '"Step 1: Every number is a multiple of 1.\\nStep 2: All {} favorable.\\nStep 3: P={}/{}={}.\\nTherefore, 1 (certain)."'),
        mk("n=__(2,10);k=__(1,n);p=k/n",
           '"Number 1..{}. P(<= {})?"',
           '"Step 1: {} numbers <= {}.\\nStep 2: P={}/{}.\\nTherefore, {}."'),
        mk("", '"Coin flipped. P(heads)?"',
           '"Step 1: 1 favorable (heads) of 2.\\nStep 2: P=1/2=0.5.\\nTherefore, 0.5."'),
    ],
    "medium": [
        mk("n=__(3,7);r=__(2,n-1);c=__.comb(n,r)",
           '"Choose {} from {}. How many ways?"',
           '"Step 1: C({},{})={}!/({}!({}-{})!).\\nStep 2: ={}.\\nTherefore, {}."'),
        mk("n=__(3,6);f=__.factorial(n)",
           '"Arrange {} distinct books."',
           '"Step 1: {}!={}.\\nTherefore, {} ways."'),
        mk("n=__(5,10);r=__(2,min(4,n-1));c=__.comb(n,r)",
           '"Committee of {} from {}?"',
           '"Step 1: C({},{})={}.\\nTherefore, {} committees."'),
        mk("n=__(4,8);r=__(2,n-2)\nc_red=__.comb(r,2)\nc_tot=__.comb(n,2)\np=round(c_red/c_tot,4)",
           '"Bag: {} red, {} blue. 2 drawn w/o replace. P(both red)?"',
           '"Step 1: Total pairs C({},2)={}. Red pairs C({},2)={}.\\nStep 2: P={}/{}.\\nTherefore, P={}."'),
        mk("p=__(1,5)/10;ans=round((1-p)**2*p,4)",
           '"P(success)={}. P(first success 3rd trial)?"',
           '"Step 1: Geometric: (1-p)^2*p.\\nStep 2: =({})^2*{}.\\nTherefore, P={}."'),
        mk("n=__(3,6);p=round((1/n)**3,5)",
           '"Die {} sides, 3 rolls. P(all 1)?"',
           '"Step 1: P(1)=1/{}.\\nStep 2: P=(1/{})^3.\\nTherefore, P={}."'),
        mk("t=__(10,20);d=__(1,3);g=t-d\np=round((g/t)*((g-1)/(t-1)),4)",
           '"{} items, {} defective. Pick 2. P(both good)?"',
           '"Step 1: P(first)={}/{}. P(second|first)={-1}/{}.\\nStep 2: P={:.4f}x{:.4f}={}.\\nTherefore, P={}."'),
        mk("pa=round(__(2,8)/10,1)\npb_a=round(__(3,9)/10,1)\npb_n=round(__(1,4)/10,1)\npn=1-pa\npb=round(pa*pb_a+pn*pb_n,4)\npa_b=round(pa*pb_a/pb,4)",
           '"P(A)={}, P(B|A)={}, P(B|~A)={}. P(A|B)?"',
           '"Step 1: Bayes: P(A|B)=P(A)P(B|A)/P(B).\\nStep 2: P(B)={}x{}+{}x{}={}.\\nStep 3: P(A|B)={}/{}={}.\\nTherefore, {}."'),
        mk("n=__(2,6);ev=(n+1)/2",
           '"{} sided die. E[X]?"',
           '"Step 1: E[X]=(1+...+{})/{}.\\nStep 2: Sum={}({}+1)/2={}.\\nStep 3: E[X]={}/{}={}.\\nTherefore, E[X]={}."'),
        mk("n=__(4,8);c=__.comb(n,2);p=round(1/c,4)",
           '"{} people. Pick 2. P(specific pair)?"',
           '"Step 1: Total pairs C({},2)={}.\\nStep 2: P=1/{}.\\nTherefore, P={}."'),
    ],
    "hard": [
        mk("", '"Permutation of n=5. P(1 before 2)?"',
           '"Step 1: Symmetry: 1 before or after 2 equally likely.\\nStep 2: P=1/2.\\nTherefore, 1/2."'),
        mk("p=__(1,9)/10;n=__(3,6);k=__(1,n-1)\nc=__.comb(n,k)\nv=round(c*p**k*(1-p)**(n-k),6)",
           '"Binomial n={}, p={}. P(X={})?"',
           '"Step 1: C({},{})={}, p^{}={:.4f}, (1-p)^{}={:.4f}.\\nStep 2: P={}x{:.4f}x{:.4f}={}.\\nTherefore, P={}."'),
        mk("n=__(3,6)", '"Roll {} die until 1. E[rolls]?"',
           '"Step 1: Geometric p=1/{}.\\nStep 2: E[X]=1/p={}.\\nTherefore, {}."'),
        mk("", '"Monty Hall: pick door 1, host opens 3. Switch?"',
           '"Step 1: P(car behind my door)=1/3.\\nStep 2: P(other door)=2/3.\\nStep 3: Host reveals goat.\\nStep 4: Switching wins 2/3.\\nTherefore, switch!"'),
        mk("n=__(3,6)\ned=(n+1)/2\neg=0.5*(2*ed)+0.5*(ed-1)\neg=round(eg,2)",
           '"Roll {} die. Flip: heads=double, tails=-1. E[X]?"',
           '"Step 1: E[die]={}.\\nStep 2: Heads:2x{}={}, Tails:{}-1={}.\\nStep 3: E=0.5x{}+0.5x{}={}.\\nTherefore, E[X]={}."'),
        mk("pa=__(1,4)/10;pb=1-pa\npu=pa+pb-pa*pb;v=round(pa/pu,4)",
           '"Two shooters: P(A)={}, P(B)={}. At least 1 hits. P(A hit)?"',
           '"Step 1: P(union)={}+{}-{}x{}={}.\\nStep 2: P(A|union)={}/{}={}.\\nTherefore, P={}."'),
        mk("n=__(5,10);k=__(2,n-2)\nct=__.comb(n,2)\ncb=__.comb(n-k,2)\ncg=ct-cb;p=round(cg/ct,4)",
           '"Hand of 2 from {} cards 1..{}. P(at least 1 <= {})?"',
           '"Step 1: Total=C({},2)={}.\\nStep 2: Both>{}=C({},2)={}.\\nStep 3: Good={}. P={}.\\nTherefore, P={}."'),
        mk("", '"Two random [0,1]. P(sum<=1)?"',
           '"Step 1: Region x+y<=1 is right triangle.\\nStep 2: Area=1/2.\\nTherefore, P=0.5."'),
        mk("n=__(4,8)\npd=1.0\nfor i in range(n):pd*=max(0,(12-i)/12)\np=round(1-pd,4)",
           '"{} people share bday month. P(at least 2 same)?"',
           '"Step 1: P(all diff)=product[0..{}] (12-i)/12={:.4f}.\\nStep 2: P(at least 2)=1-{:.4f}={}.\\nTherefore, P={}."'),
        mk("n=__(2,5)\nc=0\nfor i in range(1,n+1):\n for j in range(1,n+1):\n  if i+j==n+1:c+=1\np=round(c/(n*n),4)",
           '"Roll two {} dice. P(sum={})?"',
           '"Step 1: Total outcomes={}x{}={}.\\nStep 2: Favorable={}.\\nStep 3: P={}.\\nTherefore, P={}."'),
    ],
})

# ========= WORD PROBLEMS =========
M("word_problems", {
    "easy": [
        mk("a,b=__(10,50),__(5,20);v=a+b",
           '"Tom has {} apples. Gets {} more. Total?"',
           '"Step 1: Start={}, Gets={}.\\nStep 2: Total={}+{}={}.\\nTherefore, {}."'),
        mk("a,b=__(20,100),__(5,19);v=a-b",
           '"Train: {} passengers, {} leave. Remain?"',
           '"Step 1: {}-{}={}.\\nTherefore, {}."'),
        mk("a,b=__(2,10),__(3,10);v=a*b",
           '"{} rows, {} chairs each. Total chairs?"',
           '"Step 1: {}x{}={}.\\nTherefore, {}."'),
        mk("t=__(12,48);g=__(2,6)\nq=t//g;r=t%g",
           '"{} cookies to {} children. Each gets?"',
           '"Step 1: {}/{}={} rem {}.\\nTherefore, {} each (rem {})."'),
        mk("p=__(2,15);q=__(2,10);v=p*q",
           '"Each book ${}. Cost of {} books?"',
           '"Step 1: {}x{}={}.\\nTherefore, ${}."'),
        mk("a=__(5,30);v=2*a",
           '"Sarah {}. Father twice. Father\'s age?"',
           '"Step 1: 2x{}={}.\\nTherefore, {}."'),
        mk("s=__(10,60);t=__(1,5);d=s*t",
           '"Car {}mph for {}h. Distance?"',
           '"Step 1: {}x{}={}.\\nTherefore, {} miles."'),
        mk("l,w=__(10,50),__(5,20);a=l*w",
           '"Garden {}ftx{}ft. Area?"',
           '"Step 1: {}x{}={}.\\nTherefore, {} sq ft."'),
        mk("a,b=__(2,10),__(2,10);v=a+b",
           '"Read {} Mon, {} Tue. Total?"',
           '"Step 1: {}+{}={}.\\nTherefore, {} pages."'),
        mk("a,b=__(5,25),__(2,8);v=a*b",
           '"{} students, {} pencils each. Total?"',
           '"Step 1: {}x{}={}.\\nTherefore, {} pencils."'),
    ],
    "medium": [
        mk("t=__(20,100);ra,rb=__(1,5),__(1,5)\ntr=ra+rb;pa=t*ra//tr;pb=t-pa",
           '"Split ${} in {}:{} ratio."',
           '"Step 1: Parts={}+{}={}.\\nStep 2: Each=${}/{}={:.2f}.\\nStep 3: A={}x{:.2f}=${}.\\nStep 4: B={}x{:.2f}=${}.\\nTherefore, ${} and ${}."'),
        mk("r=__(2,8);t=__(2,6);w=r*t;v=w/r;v=int(v)if v==int(v)else v",
           '"Pipe {}gal/min to fill {}gal tank?"',
           '"Step 1: Time={}/{}={}.\\nTherefore, {} minutes."'),
        mk("p=__(10,100);d=__(10,40)\nda=round(p*d/100,2)\nfp=round(p-da,2)",
           '"Item ${}, {}% off. Sale price?"',
           '"Step 1: Discount={}%x{}={}.\\nStep 2: Sale={}-{}={}.\\nTherefore, ${}."'),
        mk("sa,sb=__(20,50),__(20,50)\nd=__(100,300)\nrs=sa+sb;t=round(d/rs,2)",
           '"Trains {}mph, {}mph, {}mi apart. Meet when?"',
           '"Step 1: Rel speed={}+{}={}.\\nStep 2: Time={}/{}={}h.\\nTherefore, {} hours."'),
        mk("a,b=__(5,20),__(2,10);v=a*b",
           '"Factory {} units/hr. {} hrs. Total?"',
           '"Step 1: {}x{}={}.\\nTherefore, {} units."'),
        mk("a,b=__(3,10),__(2,5);v=a*b+5",
           '"{} boxes, {} pencils each, +5 extra. Total?"',
           '"Step 1: {}x{}={}.\\nStep 2: +5={}.\\nTherefore, {}."'),
        mk("a,b=__(5,20),__(3,10)\np=2*(a+b);c=p*5",
           '"Rect {}x{} ft. Fence $5/ft. Cost?"',
           '"Step 1: P=2({}+{})={}.\\nStep 2: Cost={}x5=${}.\\nTherefore, ${}."'),
        mk("a,b=__(2,10),__(2,5);v=a*(2**b)",
           '"Bacteria double hourly. Start {}. After {}h?"',
           '"Step 1: 2^{}=2^{}={}.\\nStep 2: {}x{}={}.\\nTherefore, {} cells."'),
        mk("a,b=__(20,50),__(5,15);v=a-b",
           '"Alice ${}, spends ${}. Left?"',
           '"Step 1: ${}-${}=${}.\\nTherefore, ${}."'),
        mk("a=__(3,12);v=3*a",
           '"Equilateral triangle side {}. Perimeter?"',
           '"Step 1: P=3x{}={}.\\nTherefore, {}."'),
    ],
    "hard": [
        mk("ra,rb=__(1,3),__(1,3)\nc=ra+rb;t=round(1/c,2)",
           '"Pipe A fills in {}h, B in {}h. Together?"',
           '"Step 1: Rate A=1/{}={}/h, B=1/{}={}/h.\\nStep 2: Combined={}/h.\\nStep 3: Time=1/{}={}h.\\nTherefore, {} hours."'),
        mk("p=__(1000,10000);r=__(3,10);t=__(2,5)\na=round(p*(1+r/100)**t,2)\nf=round((1+r/100)**t,4)",
           '"${} at {}% compounded {}yrs. Final?"',
           '"Step 1: A=P(1+r)^t.\\nStep 2: A={}(1+{})^{}.\\nStep 3: Factor={}.\\nStep 4: A=${}x{}={}.\\nTherefore, ${}."'),
        mk("d=__(100,500);r=__(30,80);t=round(d/r,2)",
           '"Distance {} mi, speed {} mph. Time?"',
           '"Step 1: T={}/{}={}.\\nTherefore, {} hours."'),
        mk("d=__(30,80);h=__(5,15);v=d*h*5",
           '"{} widgets/hr, {}hr/day, 5 days. Total?"',
           '"Step 1: Daily={}x{}={}.\\nStep 2: 5 days={}x5={}.\\nTherefore, {} widgets."'),
        mk("a,b,c=__(2,5),__(2,5),__(2,5);v=a*b*c",
           '"Box {}x{}x{}. Unit cubes?"',
           '"Step 1: V={}x{}x{}={}.\\nTherefore, {} cubes."'),
        mk("a=__(2,6);v=a*(a-1)//2",
           '"{} people, each shakes hands with every other. Handshakes?"',
           '"Step 1: C({},2)={}x{}/2={}.\\nTherefore, {} handshakes."'),
        mk("a=__(1,5);v=a*(a+1)//2",
           '"Sum first {} positive ints?"',
           '"Step 1: {}({}+1)/2={}.\\nTherefore, {}."'),
        mk("a,b=__(10,30),__(5,15);v=a*b",
           '"Carpet ${}/sq yd, room {} sq yds. Cost?"',
           '"Step 1: {}x{}={}.\\nTherefore, ${}."'),
        mk("d=__(30,80);h=__(5,15);w=d*h",
           '"{} widgets/hr, {} hr/day, 5 days. Total?"',
           '"Step 1: Per day={}x{}={}.\\nStep 2: 5 days={}x5={}.\\nTherefore, {} widgets."'),
        mk("a,b=__(10,50),__(10,50);v=a+b",
           '"Two numbers sum to {}. One is {}. Find other."',
           '"Step 1: Other = {}-{}={}.\\nTherefore, {}."'),
    ],
})

# ========= FORMAL PROOFS =========
M("formal_proofs", {
    "easy": [
        mk("n=__(2,9);v=n*n",
           '"Prove {}^2 = {}x{}."',
           '"Step 1: {}^2 means {}x{}.\\nStep 2: {}x{}={}.\\nTherefore, {}^2={}. QED."'),
        mk("", '"Prove sum of two evens is even."',
           '"Step 1: Let a=2m, b=2n.\\nStep 2: a+b=2m+2n=2(m+n).\\nStep 3: 2(m+n) is even.\\nTherefore, sum of evens is even. QED."'),
        mk("", '"Prove product of two odds is odd."',
           '"Step 1: Let a=2m+1, b=2n+1.\\nStep 2: ab=(2m+1)(2n+1)=4mn+2m+2n+1.\\nStep 3: =2(2mn+m+n)+1.\\nStep 4: 2(2mn+m+n) is even, +1 is odd.\\nTherefore, product of odds is odd. QED."'),
        mk("", '"Prove every integer is divisible by 1."',
           '"Step 1: For any n, n/1=n (integer).\\nStep 2: So 1 divides n.\\nTherefore, every integer is divisible by 1. QED."'),
        mk("", '"Prove if x=5 then 2x=10."',
           '"Step 1: x=5 (given).\\nStep 2: 2x=2(5).\\nStep 3: 2(5)=10.\\nTherefore, 2x=10. QED."'),
        mk("", '"Prove 0 times any number is 0."',
           '"Step 1: 0xn = 0 by definition of zero.\\nStep 2: Zero added n times is 0.\\nTherefore, 0xn=0. QED."'),
        mk("", '"Prove n+0=n for any n."',
           '"Step 1: 0 is the additive identity.\\nStep 2: n+0=n.\\nTherefore, n+0=n. QED."'),
        mk("", '"Prove n+(-n)=0."',
           '"Step 1: -n is the additive inverse of n.\\nStep 2: n+(-n)=0.\\nTherefore, n+(-n)=0. QED."'),
        mk("a=__(2,9)",
           '"Prove if a={} then a+{}={}."',
           '"Step 1: a={} (given).\\nStep 2: a+{}={}+{}={}.\\nTherefore, a+{}={}. QED."'),
        mk("", '"Prove 2+3=5."',
           '"Step 1: 2+3=2+(1+1+1)=3+1+1.\\nStep 2: 3+1=4, 4+1=5.\\nTherefore, 2+3=5. QED."'),
    ],
    "medium": [
        mk("", '"Prove sqrt(2) is irrational by contradiction."',
           '"Step 1: Assume sqrt(2)=a/b in lowest terms.\\nStep 2: Square: 2=a^2/b^2 => a^2=2b^2.\\nStep 3: a^2 even => a even. Let a=2k.\\nStep 4: 4k^2=2b^2 => 2k^2=b^2 => b even.\\nStep 5: a,b both even contradicts lowest terms.\\nTherefore, sqrt(2) irrational. QED."'),
        mk("", '"Prove triangle angles sum to 180."',
           '"Step 1: Draw line through C parallel to AB.\\nStep 2: Alternate angles equal.\\nStep 3: Three angles form straight line=180.\\nTherefore, sum of angles=180. QED."'),
        mk("", '"Prove product of consecutive integers is even."',
           '"Step 1: Let n, n+1 be consecutive.\\nStep 2: If n even, n=2k => product=2k(n+1), even.\\nStep 3: If n odd, n=2k+1, n+1=2(k+1) => product=2(2k+1)(k+1), even.\\nTherefore, product of consecutive ints is even. QED."'),
        mk("", '"Prove if a|b and b|c then a|c."',
           '"Step 1: a|b => b=a*k1.\\nStep 2: b|c => c=b*k2 = a*k1*k2.\\nStep 3: k1*k2 is integer, so a|c.\\nTherefore, divisibility transitive. QED."'),
        mk("", '"Prove perfect squares cannot end in 2,3,7,8."',
           '"Step 1: Any number ends 0-9. Square each:\\nStep 2: 0^2=0, 1^2=1, 2^2=4, 3^2=9, 4^2=16->6.\\nStep 3: 5^2=25->5, 6^2=36->6, 7^2=49->9, 8^2=64->4, 9^2=81->1.\\nStep 4: Last digits: 0,1,4,5,6,9. No 2,3,7,8.\\nTherefore, QED."'),
        mk("", '"Prove between any two reals there is a rational."',
           '"Step 1: Let a<b. b-a>0.\\nStep 2: Choose n with 1/n<b-a.\\nStep 3: Let k/n<=a<(k+1)/n.\\nStep 4: (k+1)/n is rational and between a,b.\\nTherefore, rationals are dense. QED."'),
        mk("", '"Prove every integer >1 has a prime factor."',
           '"Step 1: If n is prime, n itself is a prime factor.\\nStep 2: If n composite, n=ab with a,b<n.\\nStep 3: Either a is prime or factors further.\\nStep 4: By descent, we reach a prime.\\nTherefore, every integer >1 has a prime factor. QED."'),
        mk("a=__(2,5)",
           '"Prove (x+{})^2=x^2+{}x+{}."',
           '"Step 1: (x+{})^2=(x+{})(x+{}).\\nStep 2: =x^2+{}x+{}x+{}.\\nStep 3: =x^2+{}x+{}.\\nTherefore, (x+{})^2=x^2+{}x+{}. QED."'),
        mk("a,b=__(2,5),__(2,5)",
           '"Prove (x-{})(x-{}) = x^2-{}x+{}."',
           '"Step 1: (x-{})(x-{})=x^2-{}x-{}x+{}.\\nStep 2: =x^2-({}+{})x+{}.\\nTherefore, (x-{})(x-{}) = x^2-{}x+{}. QED."'),
        mk("", '"Prove by induction: sum 1..n = n(n+1)/2."',
           '"Step 1: Base n=1: 1=1(2)/2=1.\\nStep 2: Assume true for k: sum=k(k+1)/2.\\nStep 3: For k+1: sum=k(k+1)/2+(k+1).\\nStep 4: =(k(k+1)+2(k+1))/2=(k+1)(k+2)/2.\\nStep 5: Matches formula for k+1.\\nTherefore, by induction, sum=n(n+1)/2. QED."'),
    ],
    "hard": [
        mk("", '"Prove there are infinitely many primes."',
           '"Step 1: Assume finite: p1,p2,...,pn.\\nStep 2: N=p1*p2*...*pn+1.\\nStep 3: N not divisible by any pi (rem 1).\\nStep 4: N is prime or has new prime factor.\\nStep 5: Contradiction.\\nTherefore, infinitely many primes. QED."'),
        mk("", '"Prove sqrt(p) is irrational for prime p."',
           '"Step 1: Assume sqrt(p)=a/b in lowest terms.\\nStep 2: p=a^2/b^2 => a^2=pb^2 => p|a.\\nStep 3: Let a=pk: p^2k^2=pb^2 => pk^2=b^2 => p|b.\\nStep 4: a,b share factor p, contradiction.\\nTherefore, sqrt(p) irrational. QED."'),
        mk("", '"Prove by induction: 2^n > n for all n>=1."',
           '"Step 1: Base n=1: 2^1=2>1.\\nStep 2: Assume 2^k>k for k>=1.\\nStep 3: 2^(k+1)=2*2^k>2*k (by hyp).\\nStep 4: 2*k=k+k>=k+1 (k>=1).\\nStep 5: So 2^(k+1)>k+1.\\nTherefore, by induction, 2^n>n. QED."'),
        mk("", '"Prove n^3-n is divisible by 6."',
           '"Step 1: n^3-n=n(n-1)(n+1).\\nStep 2: Three consecutive ints: n-1,n,n+1.\\nStep 3: At least one is even => divisible by 2.\\nStep 4: Exactly one divisible by 3.\\nStep 5: Divisible by both 2 and 3 => divisible by 6.\\nTherefore, n^3-n divisible by 6. QED."'),
        mk("", '"Prove by contradiction: no largest integer."',
           '"Step 1: Assume N is the largest integer.\\nStep 2: Consider N+1.\\nStep 3: N+1 > N and is integer.\\nStep 4: Contradiction.\\nTherefore, no largest integer. QED."'),
        mk("", '"Prove if ab is even, then a or b is even."',
           '"Step 1: Assume ab even but both odd.\\nStep 2: a=2m+1, b=2n+1.\\nStep 3: ab=4mn+2m+2n+1=2(2mn+m+n)+1.\\nStep 4: This is odd, contradiction.\\nTherefore, at least one factor is even. QED."'),
        mk("", '"Prove by induction: sum of squares = n(n+1)(2n+1)/6."',
           '"Step 1: Base n=1: 1=1(2)(3)/6=1.\\nStep 2: Assume for k: sum=k(k+1)(2k+1)/6.\\nStep 3: For k+1: sum=[k(k+1)(2k+1)/6]+(k+1)^2.\\nStep 4: =(k+1)[k(2k+1)+6(k+1)]/6.\\nStep 5: =(k+1)(2k^2+7k+6)/6.\\nStep 6: =(k+1)(k+2)(2k+3)/6.\\nStep 7: Matches formula for k+1.\\nTherefore, by induction, sum=n(n+1)(2n+1)/6. QED."'),
        mk("a=__(2,5)",
           '"Prove that {}! is divisible by {}."',
           '"Step 1: {}! = {}x({}-1)x...x1.\\nStep 2: {} is an explicit factor.\\nStep 3: Therefore {} divides {}!.\\nTherefore, {}|{}!. QED."'),
        mk("", '"Prove any odd prime is 4k+1 or 4k+3."',
           '"Step 1: Any integer mod 4 is 0,1,2,3.\\nStep 2: If n mod 4 = 0, n divisible by 4.\\nStep 3: If n mod 4 = 2, n is even.\\nStep 4: Odd numbers: n mod 4 = 1 or 3.\\nTherefore, odd primes are 4k+1 or 4k+3. QED."'),
        mk("", '"Prove rational numbers are countable."',
           '"Step 1: Arrange fractions in 2D grid.\\nStep 2: Diagonal traversal: (1,1),(1,2),(2,1),(1,3),(2,2),(3,1),...\\nStep 3: Every positive rational appears.\\nStep 4: Include negatives similarly.\\nTherefore, Q is countable. QED."'),
    ],
})

# Now generate the full source
print("Building math_generator.py...")

src_lines = []
def L(s=""):
    src_lines.append(s)

L("# Aurelius Training Data - Math Generator")
L("# Licensed MIT - Christien Antonio, 2026")
L('"""Synthetic math and reasoning training data generator in JSONL format.')
L()
L("Output schema:")
L('    {"instruction": str, "response": str, "category": str, "difficulty": str}')
L()
L("Categories:")
L("    arithmetic, algebra, geometry, probability, word_problems, formal_proofs")
L()
L("Difficulties:")
L("    easy   - 1-2 steps, straightforward")
L("    medium - 3-5 steps, requires multiple concepts")
L("    hard   - 5+ steps, requires insight or combination of techniques")
L('"""')
L()
L("from __future__ import annotations")
L()
L("import json")
L("import math")
L("import random")
L("from pathlib import Path")
L("from typing import Any")
L()

# Class definition
L()
L("class MathDataGenerator:")
L('    """Generates synthetic math/reasoning training data with step-by-step solutions."""')
L()
L("    CATEGORIES = [")
L('        "arithmetic",')
L('        "algebra",')
L('        "geometry",')
L('        "probability",')
L('        "word_problems",')
L('        "formal_proofs",')
L("    ]")
L('    DIFFICULTIES = ["easy", "medium", "hard"]')
L()

L("    def __init__(self, config: dict[str, Any]) -> None:")
L("        self.config = config")
L('        seed = config.get("seed", 42)')
L("        self.rng = random.Random(seed)")

def emit_generate_problem():
    L()
    L("    def generate_problem(self, category: str, difficulty: str) -> dict[str, str]:")
    L('        if category not in self.CATEGORIES:')
    L('            msg = f"Unknown category: {category}. Choose from {self.CATEGORIES}"')
    L("            raise ValueError(msg)")
    L('        if difficulty not in self.DIFFICULTIES:')
    L('            msg = f"Unknown difficulty: {difficulty}. Choose from {self.DIFFICULTIES}"')
    L("            raise ValueError(msg)")
    L()
    L('        method_name = f"_generate_{category}"')
    L("        method = getattr(self, method_name)")
    L("        instruction, response = method(difficulty)")
    L("        return {")
    L('            "instruction": instruction,')
    L('            "response": response,')
    L('            "category": category,')
    L('            "difficulty": difficulty,')
    L("        }")
    L()

emit_generate_problem()

def emit_run():
    L("    def run(self, num_samples: int, output_dir: str | Path) -> None:")
    L("        output_dir = Path(output_dir)")
    L('        math_dir = output_dir / "math"')
    L("        math_dir.mkdir(parents=True, exist_ok=True)")
    L()
    L('        config_math = self.config.get("math", {})')
    L("        type_weights = config_math.get(")
    L('            "types",')
    L("            {")
    L('                "arithmetic": 0.20,')
    L('                "algebra": 0.25,')
    L('                "geometry": 0.10,')
    L('                "probability": 0.15,')
    L('                "word_problems": 0.20,')
    L('                "formal_proofs": 0.10,')
    L("            },")
    L("        )")
    L("        diff_weights = config_math.get(")
    L('            "difficulty",')
    L('            {"easy": 0.25, "medium": 0.50, "hard": 0.25},')
    L("        )")
    L()
    L("        categories = list(type_weights.keys())")
    L("        cat_weights = [type_weights[c] for c in categories]")
    L("        diffs = list(diff_weights.keys())")
    L("        diff_w = [diff_weights[d] for d in diffs]")
    L()
    L("        records: list[dict[str, str]] = []")
    L("        for _ in range(num_samples):")
    L('            cat = self.rng.choices(categories, weights=cat_weights, k=1)[0]')
    L('            dif = self.rng.choices(diffs, weights=diff_w, k=1)[0]')
    L("            records.append(self.generate_problem(cat, dif))")
    L()
    L("        split = int(len(records) * 0.9)")
    L("        train_records = records[:split]")
    L("        val_records = records[split:]")
    L()
    L('        with open(math_dir / "train.jsonl", "w") as f:')
    L("            for rec in train_records:")
    L('                f.write(json.dumps(rec) + "\\n")')
    L()
    L('        with open(math_dir / "val.jsonl", "w") as f:')
    L("            for rec in val_records:")
    L('                f.write(json.dumps(rec) + "\\n")')
    L()

emit_run()

# Now emit each _generate_* method
for method_name, templates_by_diff in T:
    L()
    L(f'    def _generate_{method_name}(self, difficulty: str) -> tuple[str, str]:')
    L("        rng = self.rng")
    
    for diff_idx, diff in enumerate(["easy", "medium", "hard"]):
        tmpls = templates_by_diff.get(diff, [])
        
        if diff_idx == 0:
            L(f'        if difficulty == "{diff}":')
        elif diff_idx == 1:
            L(f'        elif difficulty == "{diff}":')
        else:
            L(f'        else:')
        
        L(f'            import random as _r')
        L(f'            t = _r.randint(0, {len(tmpls)-1})')
        
        for idx, (compute, inst_fmt, resp_fmt) in enumerate(tmpls):
            if idx == 0:
                L(f'            if t == 0:')
            else:
                L(f'            elif t == {idx}:')
            
            # Process compute code: replace __ with rng.
            compute_clean = compute.replace("__", "rng")
            for line in compute_clean.split("\n"):
                L(f'                {line}')
            
            L(f'                return {inst_fmt}, {resp_fmt}')
        
        # Remove the import random line that was just for the t assignment
        # Actually, let me restructure - use rng directly
    
    # We need to fix the approach above. The issue is the import random as _r line.
    # Let me do it differently - just use rng.randint directly in each template.
    L()

# The above approach is wrong - mixing code generation with computation. 
# Let me use a different strategy: store a list of (compute_fn, inst_fn, resp_fn) per diff,
# where compute_fn sets local vars, and inst_fn/resp_fn use those vars.

print(f"Generated {len(src_lines)} source lines")
print("Writing file...")

Path("/Users/christienantonio/Desktop/Aurelius/training_data/math_generator.py").write_text("\n".join(src_lines[:50]))
print("Wrote first 50 lines (test)")
print(f"Total templates: {sum(len(v) for d in T for v in d[1].values())}")
